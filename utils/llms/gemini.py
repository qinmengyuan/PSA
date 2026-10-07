import os
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
try:
    from google import genai
except ImportError:
    genai = None
try:
    from google.genai import types
except ImportError:
    types = None
import logging
try:
    import google.genai.errors
except ImportError:
    pass
import logging
import json
from datetime import datetime
import traceback
from proactive_defence.utils.format_convert import normalize_content_input
api_logger = logging.getLogger("api_interactions")
api_logger.setLevel(logging.INFO)
fh = logging.FileHandler("api_interactions.log", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
api_logger.addHandler(fh)

def safe_extract_response_fields(response):
    """Extract key fields from the response; fall back to the error message so logging never fails."""
    data = {}
    try:
        data['response_text'] = getattr(response, "text", None)
    except Exception as e:
        data['response_text'] = f"<extraction failed: {e}>"

    try:
        if hasattr(response, "candidates") and response.candidates:
            data['response_candidates'] = []
            for c in response.candidates:
                try:
                    text_part = c.content.parts[0].text
                except Exception as e:
                    text_part = f"<extraction failed: {e}>"
                data['response_candidates'].append(text_part)
        else:
            data['response_candidates'] = None
    except Exception as e:
        data['response_candidates'] = f"<extraction failed: {e}>"

    try:
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            data['total_token_count'] = getattr(usage_meta, "total_token_count", None)
        else:
            data['total_token_count'] = None
    except Exception as e:
        data['total_token_count'] = f"<extraction failed: {e}>"

    try:
        data['http_status'] = getattr(response, "status_code", None)
    except Exception as e:
        data['http_status'] = f"<extraction failed: {e}>"

    try:
        data['http_headers'] = getattr(response, "headers", None)
    except Exception as e:
        data['http_headers'] = f"<extraction failed: {e}>"

    return data


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvalidAPIResponse(Exception):
    """Exception raised to flag invalid API responses."""
    pass


class GeminiChater:
    """Gemini API rate-limit manager with multi-key rotation."""

    # Models that support the thinking_budget parameter
    THINKING_MODELS = {
        "gemini-2.5-flash-lite-preview-06-17",
        # "gemini-2.5-pro"
        # "gemini-2.0-flash"
    }

    def __init__(self, api_keys: List[str], model: str = "gemini-2.5-flash-lite-preview-06-17"):
        """
        Initialize the rate limiter.

        Args:
            api_keys: list of API keys
            model: model name to use
        """
        self.api_keys = api_keys
        self.model = model
        self.current_key_index = 0
        self.lock = threading.Lock()

        # Set per-model limits
        if "flash-lite-preview-06-17" in model.lower():
            self.rpm_limit = 15  # requests per minute
            self.tpm_limit = 250000  # tokens per minute
            self.rpd_limit = 1000  # requests per day
        elif "2.0-flash" in model.lower():
            self.rpm_limit = 15
            self.tpm_limit = 1000000  # 1,000,000 TPM
            self.rpd_limit = 200
        elif "flash" in model.lower() and "pro" not in model.lower():
            self.rpm_limit = 10
            self.tpm_limit = 250000
            self.rpd_limit = 250
        elif "pro" in model.lower():
            self.rpm_limit = 5
            self.tpm_limit = 250000
            self.rpd_limit = 100
        else:
            # Fall back to the most conservative limits
            self.rpm_limit = 5
            self.tpm_limit = 250000
            self.rpd_limit = 100

        # Track usage stats per API key
        self.key_stats = {}
        for key in api_keys:
            self.key_stats[key] = {
                'requests_minute': [],
                'tokens_minute': [],
                'requests_day': [],
                'client': genai.Client(api_key=key),
                'blocked_until': None
            }

        logger.info(f"Initialized with {len(api_keys)} API keys, model: {model}")
        logger.info(
            f"Limits: {self.rpm_limit} RPM, {self.tpm_limit} TPM, {self.rpd_limit} RPD")

    def _clean_old_records(self, key: str):
        """Purge expired usage records."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)

        # Purge per-minute records
        self.key_stats[key]['requests_minute'] = [
            t for t in self.key_stats[key]['requests_minute'] if t > minute_ago
        ]
        self.key_stats[key]['tokens_minute'] = [
            (t, tokens) for t, tokens in self.key_stats[key]['tokens_minute'] if t > minute_ago
        ]

        # Purge per-day records
        self.key_stats[key]['requests_day'] = [
            t for t in self.key_stats[key]['requests_day'] if t > day_ago
        ]

    def _can_make_request(self, key: str, estimated_tokens: int = 1000) -> bool:
        """Check whether the given key may issue a request."""
        self._clean_old_records(key)
        stats = self.key_stats[key]
        # Check whether the key is currently blocked
        if stats.get('blocked_until'):
            if datetime.now() < stats['blocked_until']:
                return False
            else:
                stats['blocked_until'] = None  # unblock

        # Check the per-minute request limit
        if len(stats['requests_minute']) >= self.rpm_limit:
            return False

        # Check the per-day request limit
        if len(stats['requests_day']) >= self.rpd_limit:
            return False

        # Check the per-minute token limit
        current_tokens = sum(tokens for _, tokens in stats['tokens_minute'])
        if current_tokens + estimated_tokens > self.tpm_limit:
            return False

        return True

    def _get_next_available_key(self, estimated_tokens: int = 1000) -> Optional[str]:
        """Return the next available API key."""
        with self.lock:
            # Scan all keys starting from the current index
            for i in range(len(self.api_keys)):
                key_index = (self.current_key_index + i) % len(self.api_keys)
                key = self.api_keys[key_index]

                if self._can_make_request(key, estimated_tokens):
                    self.current_key_index = key_index
                    return key

            return None

    def _record_usage(self, key: str, tokens_used: int):
        """Record API usage."""
        now = datetime.now()
        with self.lock:
            stats = self.key_stats[key]
            stats['requests_minute'].append(now)
            stats['requests_day'].append(now)
            stats['tokens_minute'].append((now, tokens_used))

    def _wait_for_rate_limit_reset(self):
        """Wait for the earliest unblock time among all keys."""
        now = datetime.now()
        # Collect unblock times of all blocked keys
        blocked_times = [
            stats['blocked_until'] for stats in self.key_stats.values()
            if stats['blocked_until'] and stats['blocked_until'] > now
        ]
        if blocked_times:
            min_wait = min((bt - now).total_seconds() for bt in blocked_times)
            logger.info(f"All API keys are blocked; waiting {int(min_wait)}s...")
            time.sleep(max(1, int(min_wait)))
        else:
            # No explicit unblock time; default to 10s
            logger.info("All API keys are rate-limited; waiting 60s...")
            time.sleep(60)

    def log_api_interaction(self, *, attempt, api_key, contents, system_instruction, thinking_budget,
                            success, response=None, error=None, duration=None):
        record = {
            "timestamp": datetime.now().isoformat(),
            "attempt": attempt,
            "api_key": api_key[-8:] + "...",
            "input": {
                "contents": contents,
                "system_instruction": system_instruction,
                "thinking_budget": thinking_budget,
            },
            "success": success,
            "duration_seconds": duration,
            "error": str(error) if error else None,
        }

        if response:
            record.update(safe_extract_response_fields(response))

        api_logger.info(json.dumps(record, ensure_ascii=False))

    def generate_content(self,
                         contents: str,
                         system_instruction: Optional[str] = None,
                         thinking_budget: Optional[int] = 0,
                         max_retries: int = 3,
                         history=None) -> Dict[str, Any]:
        """
        Generate content; history input is handled via create-chat + send.

        Args:
            contents: input content (a string or a list of history messages)
            system_instruction: system instruction
            thinking_budget: thinking budget (only for specific models)
            max_retries: maximum number of retries

        Returns:
            API response dict
        """

        # Estimate token count for history input
        if history:
            estimated_tokens = len(str(history).split()) * 2
        else:
            estimated_tokens = len(contents.split()) * 2  # rough token estimate

        start_time = time.time()
        attempt = 0
        while attempt < max_retries:
            available_key = self._get_next_available_key(estimated_tokens)
            if not available_key:
                logger.warning("No API key available; waiting for unblock...")
                self._wait_for_rate_limit_reset()
                continue  # do not increment attempt
            try:
                # Build the request config
                config_params = {}

                # Add the system instruction
                if system_instruction:
                    config_params['system_instruction'] = system_instruction

                # Add thinking budget (specific models only)
                if thinking_budget is not None and self.model in self.THINKING_MODELS:
                    config_params['thinking_config'] = types.ThinkingConfig(
                        thinking_budget=thinking_budget)

                config = types.GenerateContentConfig(
                    **config_params) if config_params else None

                # Send the request
                client = self.key_stats[available_key]['client']

                if history:
                    # History mode: create a chat session and send the message
                    logger.info(f"Using API key {available_key[-8:]}... creating chat session")
                    chat = client.chats.create(
                        model=self.model,
                        config=config,
                        history=[normalize_content_input(i) for i in history]
                    )
                    logger.info(f"Using API key {available_key[-8:]}... sending history message")
                    response = chat.send_message(contents)
                else:
                    # Plain mode: generate content directly
                    logger.info(f"Using API key {available_key[-8:]}... sending request")
                    response = client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config
                    )

                duration = time.time() - start_time
                self.log_api_interaction(
                    attempt=attempt + 1,
                    api_key=available_key,
                    contents=str(contents),
                    system_instruction=system_instruction,
                    thinking_budget=thinking_budget,
                    success=True,
                    response=response,
                    duration=duration
                )

                try:
                    if response is None:
                        raise InvalidAPIResponse("API returned an empty response (None)")

                    # Check for the text or candidates attribute
                    if not hasattr(response, 'text') and not hasattr(response, 'candidates'):
                        raise InvalidAPIResponse(
                            "Invalid API response: missing 'text' or 'candidates' attribute")

                    # Check whether candidates is empty
                    if hasattr(response, 'candidates') and not response.candidates:
                        raise InvalidAPIResponse("No candidates in API response")
                except InvalidAPIResponse as e:
                    print(f"[ERROR] API response validation failed: {e}")
                    if response.prompt_feedback.block_reason == 'PROHIBITED_CONTENT':
                        logger.error("API flagged prohibited content; the content may violate usage policies")
                        return {'response': "CUSTOM_ERROR: PROHIBITED_CONTENT", 'text': ''}

                # Extract the response text
                response_text = ""
                try:
                    response_text = response.text if hasattr(
                        response, 'text') else ""
                    if response_text == "":
                        raise InvalidAPIResponse("API returned an empty text string")
                except Exception as e:
                    logger.warning(f"Failed to get response text: {e}")
                    if hasattr(response, 'candidates') and response.candidates:
                        # Try extracting text from candidates
                        try:
                            response_text = response.candidates[0].content.parts[0].text
                        except:
                            response_text = "<unable to extract response text>"

                # Record usage
                # Read actual token usage from the response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    total_tokens = getattr(
                        response.usage_metadata, 'total_token_count', estimated_tokens)
                else:
                    total_tokens = estimated_tokens

                self._record_usage(available_key, total_tokens)

                logger.info(f"Request succeeded, used {total_tokens} tokens")
                if not response_text:
                    logger.warning(f"Empty response text; full API response: {response}")
                # Return the full response object and usage stats
                return {
                    'response': response,
                    'text': response_text,
                    'api_key_used': available_key[-8:] + "...",
                    'tokens_used': total_tokens,
                    'attempt': attempt + 1
                }

            except google.genai.errors.ClientError as ce:
                # Check for a 400 INVALID_ARGUMENT error
                if getattr(ce, 'status', None) == 'INVALID_ARGUMENT' and 'unexpected model name format' in str(ce):
                    logger.error(f"Invalid model name: {ce}")
                    raise ce  # re-raise without retrying

                duration = time.time() - start_time
                self.log_api_interaction(
                    attempt=attempt + 1,
                    api_key=available_key,
                    contents=str(contents),
                    system_instruction=system_instruction,
                    thinking_budget=thinking_budget,
                    success=False,
                    error=ce,
                    duration=duration
                )

                wait_seconds = 60

                if (getattr(ce, 'status', None) == 'RESOURCE_EXHAUSTED') and (details := getattr(ce, 'details', None).get('error', None).get('details', None)):
                    details = {
                        key: value
                        for detail in details
                        if (attrtype := detail.get('@type')) and (
                            (attrtype == 'type.googleapis.com/google.rpc.QuotaFailure' and (key := 'quotaId') and (value := detail.get('violations', [{}])[0].get('quotaId'))) or
                            (attrtype == 'type.googleapis.com/google.rpc.RetryInfo' and (key :=
                                                                                         'retryDelay') and (value := detail.get('retryDelay')))
                        )
                    }

                    if details.get('quotaId') == 'GenerateRequestsPerDayPerProjectPerModel-FreeTier':
                        tomorrow = datetime.now() + timedelta(days=1)
                        blocked_until = datetime(
                            tomorrow.year, tomorrow.month, tomorrow.day, 1, 1, 0)
                        self.key_stats[available_key]['blocked_until'] = blocked_until
                        logger.error(
                            f"API key {available_key[-8:]}... daily quota exhausted; unavailable until midnight")
                        continue
                    elif details.get('quotaId') == 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier':
                        time.sleep(62)
                        logger.error(
                            f"API key {available_key[-8:]}... per-minute quota exhausted; pausing for one minute")
                    elif wait_seconds := int(details.get('retryDelay')[:-1]):
                        self.key_stats[available_key]['blocked_until'] = datetime.now(
                        ) + timedelta(seconds=wait_seconds)
                        logger.error(str(ce))
                        logger.error(
                            f"API key {available_key[-8:]}... resource exhausted; available in {wait_seconds}s")  # try the next available key directly
                        continue

                logger.error(f"ClientError: {ce}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise ce
            except Exception as e:
                duration = time.time() - start_time
                self.log_api_interaction(
                    attempt=attempt + 1,
                    api_key=available_key,
                    contents=str(contents),
                    system_instruction=system_instruction,
                    thinking_budget=thinking_budget,
                    success=False,
                    error=e,
                    duration=duration
                )
                traceback.print_exc() 
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                attempt += 1
                continue
        raise Exception("All retries failed; please try again later")

    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return usage stats for all API keys."""
        stats = {}
        for key in self.api_keys:
            self._clean_old_records(key)
            key_display = key[-8:] + "..."

            current_tokens = sum(
                tokens for _, tokens in self.key_stats[key]['tokens_minute'])

            stats[key_display] = {
                'requests_this_minute': len(self.key_stats[key]['requests_minute']),
                'requests_today': len(self.key_stats[key]['requests_day']),
                'tokens_this_minute': current_tokens,
                'rpm_remaining': max(0, self.rpm_limit - len(self.key_stats[key]['requests_minute'])),
                'rpd_remaining': max(0, self.rpd_limit - len(self.key_stats[key]['requests_day'])),
                'tpm_remaining': max(0, self.tpm_limit - current_tokens)
            }

        return stats


# Usage example
if __name__ == "__main__":
    # Configure your API key list
    # Never hardcode API keys. Provide them via the GEMINI_API_KEYS
    # environment variable (comma-separated) or utils/keys_loader.py.
    API_KEYS = [k for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k]

    # Create the rate limiter (multiple models supported)
    # Available models:
    # - "gemini-2.5-flash-lite-preview-06-17" (15 RPM, 250K TPM, 1000 RPD)
    # - "gemini-2.0-flash" (15 RPM, 1M TPM, 200 RPD)
    # - "gemini-2.5-flash" (10 RPM, 250K TPM, 250 RPD)
    # - "gemini-2.5-pro" (5 RPM, 250K TPM, 100 RPD)
    limiter = GeminiChater(
        api_keys=API_KEYS,
        model="gemini-2.0-flash"  # or any other supported model
    )

    try:
        # Example 1: basic call
        result1 = limiter.generate_content(
            contents="Explain how AI works in a few words",
            thinking_budget=0  # disable thinking mode
        )
        print("Example 1 response:", result1['text'])
        print("API key used:", result1['api_key_used'])
        print("Tokens used:", result1['tokens_used'])

        # Example 2: call with a system instruction
        result2 = limiter.generate_content(
            contents="Hello there",
            system_instruction="You are a cat. Your name is Neko."
        )
        print("\nExample 2 response:", result2['text'])

        # Example 3: burst calls to exercise rate limiting
        print("\nStarting burst-call test...")
        for i in range(5):
            result = limiter.generate_content(
                contents=f"Tell me a fun fact about number {i+1}",
                thinking_budget=0
            )
            print(f"Call {i+1}: {result['text'][:50]}...")
            time.sleep(1)  # short delay

        # Show usage stats
        print("\nCurrent usage stats:")
        stats = limiter.get_usage_stats()
        for key, stat in stats.items():
            print(f"API key {key}:")
            print(
                f"  Requests this minute: {stat['requests_this_minute']}/{limiter.rpm_limit}")
            print(f"  Requests today: {stat['requests_today']}/{limiter.rpd_limit}")
            print(
                f"  Tokens this minute: {stat['tokens_this_minute']}/{limiter.tpm_limit}")
            print()

    except Exception as e:
        logger.error(f"Error: {str(e)}")

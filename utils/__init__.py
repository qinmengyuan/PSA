
import yaml
# Load configuration from YAML
def load_config(path="/data/psa/khgfds/proactive_defence/train/qwen_based/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../')))
from proactive_defence import proactiveDefencer
from proactive_defence.temp_llm import LazyMetaLlama
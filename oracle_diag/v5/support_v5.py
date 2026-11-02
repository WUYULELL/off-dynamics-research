import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v4'))
from support_v4 import oracle_receipt, model_hash, array_hash
from common_v5 import *

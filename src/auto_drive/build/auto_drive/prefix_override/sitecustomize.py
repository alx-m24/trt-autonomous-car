import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jackerost/autonomous_prototype/src/auto_drive/install/auto_drive'

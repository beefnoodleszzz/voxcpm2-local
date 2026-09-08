# Voice Library

Only use voices you own or have permission to clone. Add a reference with:

```bash
.venv/bin/python scripts/create_voice.py --id male_lead_01 --style neutral \
  --reference /absolute/path/reference.wav --transcript "逐字准确文本"
```

The original file is preserved alongside a 48kHz mono PCM-24 model copy. Neither is overwritten.


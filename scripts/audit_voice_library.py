#!/usr/bin/env python3
"""Verify every registered voice style and its exact transcript."""
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from src.audio_utils import inspect_audio
from src.voice_library import VoiceLibrary

def main():
    lib=VoiceLibrary(); errors=[]; warnings=[]; styles=0; report=[]
    for voice in lib.list():
        for style in voice.get('styles',{}):
            styles+=1
            try:
                entry=lib.resolve_style(voice['id'],style); stats=inspect_audio(entry['audio'])
                if not entry['transcript']: errors.append(f"{voice['id']}/{style}: empty transcript")
                if stats['sample_rate']!=48000 or stats['channels']!=1 or stats['has_nan'] or stats['clipping']:
                    errors.append(f"{voice['id']}/{style}: invalid audio {stats}")
                if stats['duration'] < 5.0:
                    warnings.append(f"{voice['id']}/{style}: reference is {stats['duration']:.2f}s; use 5-30s, preferably 8-15s")
                if stats['peak'] > 0.98:
                    warnings.append(f"{voice['id']}/{style}: peak {stats['peak']:.3f} is close to full scale")
                report.append({'character_id':voice['id'],'name':voice['name'],'style':style,
                               'duration':stats['duration'],'peak':stats['peak'],'rms':stats['rms']})
            except Exception as exc: errors.append(f"{voice['id']}/{style}: {exc}")
    summary={'characters':len(lib.list()),'styles':styles,'errors':errors,'warnings':warnings,'entries':report}
    Path('voices/audit_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'characters':summary['characters'],'styles':styles,'errors':errors,
                      'warning_count':len(warnings)},ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)
if __name__=='__main__': main()

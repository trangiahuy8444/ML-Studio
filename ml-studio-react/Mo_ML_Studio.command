#!/bin/zsh
ml_studio_file="${0:A:h}/ML_Studio_React.html"
if [[ -d '/Applications/Google Chrome.app' ]]; then
  open -a 'Google Chrome' "$ml_studio_file"
else
  open "$ml_studio_file"
fi

#!/bin/bash
pandoc -t html5 \
  --template=/home/albertauyeung/talks/deep-learning/template.html \
  --standalone --section-divs \
  --variable theme="custom" \
  --variable transition="none" \
  /home/albertauyeung/talks/deep-learning/index.md \
  -o /home/albertauyeung/talks/deep-learning/index.html


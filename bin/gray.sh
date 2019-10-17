#!/bin/sh
for f in `find . -name '*.png'`; do
    convert ${f} -type GrayScale ${f}-g
    if [ -f ${f}-g ]; then
	mv -f ${f}-g ${f}
    fi
    let i++
done

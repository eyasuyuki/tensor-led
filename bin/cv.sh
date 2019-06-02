
#!/bin/sh
path="./images/"
ext=".png"
pics=("n0" "n1" "n2" "n3" "n4" "n5" "n6" "n6b" "n7" "n7b" "n8" "n9")
for e in ${pics[@]}; do
    # rotate +5 r5
    convert -background black -rotate +5 ${path}${e}${ext} ${path}${e}-r5${ext}
    # rotate -5 l5
    convert -background black -rotate -5 ${path}${e}${ext} ${path}${e}-l5${ext}
    # blur 3
    convert -background black -blur 3x3 ${path}${e}${ext} ${path}${e}-b3${ext}
    convert -background black -blur 3x3 ${path}${e}-r5${ext} ${path}${e}-r5-b3${ext}
    convert -background black -blur 3x3 ${path}${e}-l5${ext} ${path}${e}-l5-b3${ext}
    # blur 6
    convert -background black -blur 6x6 ${path}${e}${ext} ${path}${e}-b6${ext}
    convert -background black -blur 6x6 ${path}${e}-r5${ext} ${path}${e}-r5-b6${ext}
    convert -background black -blur 6x6 ${path}${e}-l5${ext} ${path}${e}-l5-b6${ext}
    let i++
done

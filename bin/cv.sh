
#!/bin/sh
path="./images/"
ext=".png"
pics=("0/n0" "1/n1" "2/n2" "3/n3" "4/n4" "5/n5" "6/n6" "6/n6b" "7/n7" "7/n7b" "8/n8" "9/n9")
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

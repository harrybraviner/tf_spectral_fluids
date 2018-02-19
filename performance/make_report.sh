#! /bin/bash

FILENAME="./report.html"

FIRST=1

for DIR in `ls ./`; do
    if [ -d $DIR ]; then
        if [ $FIRST -eq 1 ]; then
            markdown $DIR/report.md | sed "s/\.\//.\/$DIR\//g" > $FILENAME
        else
            markdown $DIR/report.md >> $FILENAME
        fi
    fi
done

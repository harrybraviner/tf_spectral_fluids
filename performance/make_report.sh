#! /bin/bash

FILENAME="./report.html"

FIRST=1

DIRS=`ls -1 ./`

for DIR in $DIRS; do
    if [ -d $DIR ]; then
        if [ $FIRST -eq 1 ]; then
            markdown $DIR/report.md | sed "s/\.\//.\/$DIR\//g" > $FILENAME
            FIRST=0
        else
            markdown $DIR/report.md | sed "s/\.\//.\/$DIR\//g" >> $FILENAME
        fi
    fi
done

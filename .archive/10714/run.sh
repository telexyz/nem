wget https://raw.githubusercontent.com/yks72p/dl_sys_course_notes/main/README.md
mv README.md.1 LECTURES_NOTE.md

cd notebooks && mv _git .git && git pull
mv .git _git && cd ..

cd hw3 && mv _git .git && git pull
mv .git _git && cd ..

cd hw4 && mv _git .git && git pull
mv .git _git && cd ..

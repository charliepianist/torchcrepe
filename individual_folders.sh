for i in {1..50}
    do
        if [ $i -ge 10 ]; then
            # Create individual folders and copy over wav files
            mkdir "data/test/CSD/wav/en0${i}a"
            cp "data/test/CSD/wav/converted/en0${i}a.wav" "data/test/CSD/wav/en0${i}a/"
            mkdir "data/test/CSD/wav/en0${i}b"
            cp "data/test/CSD/wav/converted/en0${i}b.wav" "data/test/CSD/wav/en0${i}b/"

            # Create individual folders and copy over lexicons
            mkdir "data/test/CSD/lyric/en0${i}a"
            cp "data/test/CSD/lyric/en0${i}a.txt" "data/test/CSD/lyric/en0${i}a/"
            mkdir "data/test/CSD/lyric/en0${i}b"
            cp "data/test/CSD/lyric/en0${i}b.txt" "data/test/CSD/lyric/en0${i}b/"
        fi
        if [ 9 -ge $i ]; then
            # Create individual folders and copy over wav files
            mkdir "data/test/CSD/wav/en00${i}a"
            cp "data/test/CSD/wav/converted/en00${i}a.wav" "data/test/CSD/wav/en00${i}a/"
            mkdir "data/test/CSD/wav/en00${i}b"
            cp "data/test/CSD/wav/converted/en00${i}b.wav" "data/test/CSD/wav/en00${i}b/"

            # Create individual folders and copy over lexicons
            mkdir "data/test/CSD/lyric/en00${i}a"
            cp "data/test/CSD/lyric/en00${i}a.txt" "data/test/CSD/lyric/en00${i}a/"
            mkdir "data/test/CSD/lyric/en00${i}b"
            cp "data/test/CSD/lyric/en00${i}b.txt" "data/test/CSD/lyric/en00${i}b/"
        fi
    done
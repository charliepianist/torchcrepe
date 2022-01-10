for i in {1..50}
    do
        if [ $i -ge 10 ]; then
            sox "data/test/CSD/wav/en0${i}a.wav" "data/test/CSD/wav/converted/en0${i}a.wav"
            sox "data/test/CSD/wav/en0${i}b.wav" "data/test/CSD/wav/converted/en0${i}b.wav"
        fi
        if [ 9 -ge $i ]; then
            sox "data/test/CSD/wav/en00${i}a.wav" "data/test/CSD/wav/converted/en00${i}a.wav"
            sox "data/test/CSD/wav/en00${i}b.wav" "data/test/CSD/wav/converted/en00${i}b.wav"
        fi
    done
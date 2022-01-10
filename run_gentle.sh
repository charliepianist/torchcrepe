if [[ $1 -eq 0 ]] ; then
    echo 'Usage: bash scripts/run_gentle.sh <port of gentle docker container>'
    exit 0
fi

for i in {1..50}
    do
        if [ $i -ge 10 ]; then
            # Output JSON
            curl -X POST -F "audio=@data/test/CSD/wav/en0${i}a.wav" -F "transcript=@data/test/CSD/lyric/en0${i}a.txt" "http://localhost:${1}/transcriptions?async=false" > "out/align/en0${i}a.json"
            curl -X POST -F "audio=@data/test/CSD/wav/en0${i}b.wav" -F "transcript=@data/test/CSD/lyric/en0${i}b.txt" "http://localhost:${1}/transcriptions?async=false" > "out/align/en0${i}b.json"
        fi
        if [ 9 -ge $i ]; then
            # Output JSON
            curl -X POST -F "audio=@data/test/CSD/wav/en00${i}a.wav" -F "transcript=@data/test/CSD/lyric/en00${i}a.txt" "http://localhost:${1}/transcriptions?async=false" > "out/align/en00${i}a.json"
            curl -X POST -F "audio=@data/test/CSD/wav/en00${i}b.wav" -F "transcript=@data/test/CSD/lyric/en00${i}b.txt" "http://localhost:${1}/transcriptions?async=false" > "out/align/en00${i}b.json"
        fi
    done
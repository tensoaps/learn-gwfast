thread=20
tmp_fifofile=/tmp/$$.fifo
mkfifo $tmp_fifofile
exec 8<>$tmp_fifofile
rm $tmp_fifofile

for i in `seq $thread`
do
    echo >&8
done

for j in $(seq 50)
do
    read -u 8
    {
        python gwfast_demo.py -run $j
        echo >&8
    }&
done

wait
exec 8>&-

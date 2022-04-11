
trap signalExit 2
function signalExit(){
    echo "\r"
    kill $(jobs -p)
    exit 2
}

if [ $# -lt 5 ]
then
    echo "argument error: usage: bash $0 dir N LENGTH WIDTH PADDING SUFFIX"
    exit 1
fi
SUFFIX=""
if [ $# -gt 5 ]
then
    SUFFIX=$6
fi

DIR=$1
N=$2
LENGTH=$3
WIDTH=$4
PADDING=$5

WIDTH_P=`echo $(echo $WIDTH | awk '{print $1 * 1000}')`
WIDTH_P=`printf %04d $WIDTH_P`

for I in `seq 1 $N`
do
    I_P=`printf %02d $I`
    bash auto_quantification.sh $DIR/$I_P $LENGTH $WIDTH $SUFFIX &
done
wait

OUTFILE="$DIR/fig_ifr_W${WIDTH_P}_${LENGTH}_${PADDING}.pdf"
if [ ! -e $OUTFILE ]
then
python plot_ifr_multi.py $DIR/*/ifr_W${WIDTH_P}_*.npz -o $OUTFILE --times $LENGTH --padding $PADDING
fi
OUTFILE="$DIR/fig_slr_W${WIDTH_P}_${LENGTH}_${PADDING}.pdf"
if [ ! -e $OUTFILE ]
then
python plot_slr_multi.py $DIR/*/slr_W${WIDTH_P}_*.npz -o $OUTFILE --times $LENGTH --padding $PADDING
fi


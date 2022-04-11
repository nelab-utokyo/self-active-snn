
trap signalExit 2
function signalExit(){
    echo "\r"
    kill $(jobs -p)
    exit 2
}

if [ $# -ne 5 ]
then
    echo "argument error: usage: bash $0 DIR N LENGTH WIDTH PADDING"
    exit 1
fi

DIR=$1
N=$2
LENGTH=$3
WIDTH=$4
PADDING=$5

for I in `seq 1 $N`
do
    I_P=`printf %02d $I`
    bash auto_quantification_development_xN.sh $DIR/$I_P $LENGTH $WIDTH $PADDING &
done
wait

WIDTH_P=`echo $(echo $WIDTH | awk '{print $1 * 1000}')`
WIDTH_P=`printf %04d $WIDTH_P`

OUTFILE="$DIR/fig_ifr_W${WIDTH_P}.pdf"
if [ ! -e $OUTFILE ]
then
python plot_ifr_multi.py $DIR/*/ifr_new_W${WIDTH_P}_*.npz -o $OUTFILE -l 0.32
fi
OUTFILE="$DIR/fig_slr_W${WIDTH_P}.pdf"
if [ ! -e $OUTFILE ]
then
python plot_slr_multi.py $DIR/*/slr_new_W${WIDTH_P}_*.npz -o $OUTFILE -l 0.2
fi


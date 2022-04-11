
trap signalExit 2
function signalExit(){
    echo "\r"
    kill $(jobs -p)
    exit 2
}

if [ $# -ne 5 ]
then
    echo "argument error: usage: bash $0 config outdir padding length pattern"
    exit 1
fi

if [ ! -e $1 ]
then
    echo "error: $1 does not exist"
    exit 1
elif [ ! -d $2 ]
then
    mkdir $2
elif [ ! -e $5 ]
then
    echo "error: $5 does not exist"
    exit 1
fi

C_FILE=$1
BASEDIR=$2
PADDING=$3
LENGTH=$4
PATTERN_FILE=$5

echo -e "c_file=$C_FILE\ntime=$TIME\npattern_file=$PATTERN_FILE" > $BASEDIR/setting.txt

S_FILES=`echo $C_FILE | sed -e "s/config/states/" | sed -e "s/\.json/_time0000_\*/"`
S_FILES=`ls $S_FILES`
for S_FILE in ${S_FILES[@]}
do
    BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
    BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
    SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
    DIR="$BASEDIR/$SEED_NUM"

    if [ ! -d $DIR ]
    then
    mkdir $DIR
    fi
done

for TIME in `seq 0 $PADDING $LENGTH`
do
    TIME_P=`printf %04d $TIME`
    echo $TIME_P

    S_FILES=`echo $C_FILE | sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}_\*/"`
    S_FILES=`ls $S_FILES`

    for S_FILE in ${S_FILES[@]}
    do
        BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
        BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
        SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
        DIR="$BASEDIR/$SEED_NUM"

        bash spatial_input_single.sh $C_FILE $S_FILE $DIR $TIME_P \
        $PATTERN_FILE $SEED_NUM PARALLEL &
    done
    wait
done

#for S_FILE in ${S_FILES[@]}
#do
#    BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
#    BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
#    SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
#    DIR="$BASEDIR/$SEED_NUM"
#    python analyze_synapse.py $S_FILE $DIR/states_after*.npz -o $BASEDIR/synapse_$SEED_NUM.npz
#done
#
#if [ ! -e $BASEDIR/synapse.pdf ]
#then
#    python plot_development.py $BASEDIR/synapse_*.npz -o $BASEDIR/synapse.pdf
#fi


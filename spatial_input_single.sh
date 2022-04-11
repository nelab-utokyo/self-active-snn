
trap signalExit 2
function signalExit(){
    echo "\r"
    exit 2
}

REP=40

if [ $# -le 5 ]
then
    echo "argument error: usage: bash $0 c_file s_file outdir \
    time pattern SEED_NUM PARALLEL(OPTION)"
    exit 1
fi
OPTION=""
if [ $# -ge 7 ]
then
    OPTION="--parallel"
fi

if [ ! -e $1 ]
then
    echo "error: $1 does not exist"
    exit 1
elif [ ! -e $2 ]
then
    echo "error: $2 does not exist"
    exit 1
elif [ ! -d $3 ]
then
    mkdir $3
elif [ ! -e $5 ]
then
    echo "error: $5 does not exist"
    exit 1
fi

S_FILE=$2
PATTERN_FILE=$5
SEED_NUM=$6
N_PATTERNS=(`cat $PATTERN_FILE | wc -l`)

for I in `seq $N_PATTERNS`
do
    I_=`expr $I - 1`
    I_P=`printf %02d $I_`
    PATTERN=(`sed -n ${I}p $PATTERN_FILE`)
    RESULT=`ls $3/${I_P}_*_$4.npz`
    if [ $? -eq 0 ]
    then
        for R in `seq $REP`
        do
            R_=`expr $R - 1`
            SEED=`expr $R_ + $I \* 1000 + $SEED_NUM \* 1000000`
            OUTFILE=`printf %s/%02d_%03d_$4.npz $3 $I_ $R_`
            echo $OUTFILE, ${PATTERN[@]}, $SEED
            if [ ! -e $OUTFILE ]
            then
            python snn_spatial_input.py $1 $2 -o $OUTFILE -s ${PATTERN[@]} \
            --seed $SEED $OPTION
            else
            echo PASS: $OUTFILE
            fi
        done
    else
        SEED=`expr $I \* 1000 + $SEED_NUM \* 1000000`
        python snn_spatial_input_mt.py $1 $2 -o $3/$I_P -s ${PATTERN[@]} \
        --seed $SEED $OPTION --suffix _$4 -t $REP
    fi
done


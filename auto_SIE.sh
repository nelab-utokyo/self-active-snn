
trap signalExit 2
function signalExit(){
    echo "\r"
    kill $(jobs -p)
    exit 2
}

#DE="0.0001"
#DI="0.0001"
#RUNTIME=600
RUNTIME=1800

if [ $# -lt 7 ]
then
    echo "argument error: usage: bash $0 config outdir time length pattern n_stim padding ANALYSIS_ONLY(OPTION)"
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
ANALYSIS_ONLY=0
if [ $# -gt 7 ]
then
    ANALYSIS_ONLY=1
fi


C_FILE=$1
BASEDIR=$2
TIME=$3
LEN=$4
PATTERN_FILE=$5
N_STIM=$6
PADDING=$7
TIME_P=`printf %04d $TIME`

S_FILES=`echo $C_FILE | sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}_\*/"`
S_FILES=`ls $S_FILES`

BLOCKS=(`echo $C_FILE | tr '/' '\n'`)
if [ ! -e $BASEDIR/${BLOCKS[${#BLOCKS[@]}-1]} ]
then
    cp $C_FILE $BASEDIR/
fi

if [ $ANALYSIS_ONLY == 0 ]
then
echo -e "c_file=$C_FILE\ntime=$TIME\nLEN=$LEN\nN_STIM=$N_STIM\nPATTERN_FILE=$PATTERN_FILE" > $BASEDIR/setting.txt
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
echo repetitive stimulation
for S_FILE in ${S_FILES[@]}
do
    BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
    BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
    SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
    DIR="$BASEDIR/$SEED_NUM"

    SEED=`expr $SEED_NUM \* 10000 + 1000`

    if [ ! -e $DIR/states_after00.npz ]
    then
    echo $DIR/states_after00.npz, $SEED
    python snn_rstim.py $C_FILE $S_FILE $DIR/states_after00.npz \
    --parallel --pattern_file $PATTERN_FILE --rep $N_STIM --seed $SEED &
    #--de $DE --di $DI --parallel --pattern_file $PATTERN_FILE &
    else
    echo PASS: $DIR/states_after00.npz
    fi
done
wait
echo spontaneous activity
for TIME_AFTER in `seq 1 $LEN`
do
    TIME_AFTER_P=`printf after%02d $TIME_AFTER`
    TIME_PRE=`expr $TIME_AFTER - 1`
    TIME_PRE_P=`printf after%02d $TIME_PRE`
    echo $TIME_AFTER_P

    for S_FILE in ${S_FILES[@]}
    do
        BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
        BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
        SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
        DIR="$BASEDIR/$SEED_NUM"

        SEED=`expr $SEED_NUM \* 10000 + $TIME_AFTER`

        S_FILE=$DIR/states_$TIME_AFTER_P.npz
        IS_FILE=$DIR/states_$TIME_PRE_P.npz

        if [ ! -e $S_FILE ]
        then
        echo $S_FILE, $SEED
        python snn_spont.py $C_FILE $S_FILE -s $IS_FILE -l 1 --parallel &
        else
        echo PASS: $S_FILE
        fi
    done
    wait
done
echo record evoked responses
echo before
for S_FILE in ${S_FILES[@]}
do
    BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
    BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
    SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
    DIR="$BASEDIR/$SEED_NUM"

    bash spatial_input_single.sh $C_FILE $S_FILE $DIR before \
    $PATTERN_FILE $SEED_NUM PARALLEL &
done
wait
for TIME_AFTER in `seq 0 $PADDING $LEN`
do
    TIME_AFTER_P=`printf after%02d $TIME_AFTER`
    echo $TIME_AFTER_P

    for S_FILE in ${S_FILES[@]}
    do
        BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
        BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
        SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
        DIR="$BASEDIR/$SEED_NUM"

        S_FILE=$DIR/states_$TIME_AFTER_P.npz
        bash spatial_input_single.sh $C_FILE $S_FILE $DIR $TIME_AFTER_P \
        $PATTERN_FILE $SEED_NUM PARALLEL &
    done
    wait
done
echo record spontaneous activity
RUNTIME_P=`printf %04d $RUNTIME`
for TIME_AFTER in `seq 0 $LEN`
do
    TIME_AFTER_P=`printf after%02d $TIME_AFTER`
    echo $TIME_AFTER_P

    for S_FILE in ${S_FILES[@]}
    do
        BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
        BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
        SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
        DIR="$BASEDIR/$SEED_NUM"

        S_FILE=$DIR/states_$TIME_AFTER_P.npz
        RECORD_FILE=$DIR/record_${RUNTIME_P}_$TIME_AFTER_P.npz
        if [ ! -e $RECORD_FILE ]
        then
            python snn_record.py $C_FILE $S_FILE $RECORD_FILE \
            -l $RUNTIME --spike_only --parallel &
        fi
    done
    wait
done
fi # ANALYSIS_ONLY == 0

for S_FILE in ${S_FILES[@]}
do
    BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
    BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
    SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
    DIR="$BASEDIR/$SEED_NUM"

    S_FILES_AFTER=()
    for TIME_AFTER in `seq 0 $PADDING $LEN`
    do
        TIME_AFTER_P=`printf after%02d $TIME_AFTER`
        S_FILES_AFTER=("${S_FILES_AFTER[@]}" "$DIR/states_$TIME_AFTER_P.npz")
    done
    OUTFILE="$BASEDIR/synapse_${SEED_NUM}_${LEN}_${PADDING}.npz"
    if [ ! -e $OUTFILE ]
    then
        python analyze_synapse.py $S_FILE ${S_FILES_AFTER[@]} -o $OUTFILE
    fi
done

for S_FILE in ${S_FILES[@]}
do
    BLOCKS=(`echo $S_FILE | tr '\.' '\n'`)
    BLOCKS=(`echo ${BLOCKS[0]} | tr '_' '\n'`)
    SEED_NUM="${BLOCKS[${#BLOCKS[@]}-1]}"
    DIR="$BASEDIR/$SEED_NUM"

    R_FILE=`echo $S_FILE | sed -e "s/states/record_${RUNTIME_P}/"`

    R_FILES_AFTER=()
    for TIME_AFTER in `seq 0 $PADDING $LEN`
    do
        TIME_AFTER_P=`printf after%02d $TIME_AFTER`
        R_FILES_AFTER=("${R_FILES_AFTER[@]}" "$DIR/record_${RUNTIME_P}_$TIME_AFTER_P.npz")
    done

    OUTFILE="$BASEDIR/indices_${SEED_NUM}_${LEN}_${PADDING}.npz"
    if [ ! -e $OUTFILE ]
    then
        python analyze_indices.py $R_FILE ${R_FILES_AFTER[@]} -o $OUTFILE -n 100
    fi
done

#OUTFILE="$BASEDIR/fig_synapse_${LEN}_${PADDING}.pdf"
#if [ ! -e $OUTFILE ]
#then
#    python plot_development.py $BASEDIR/synapse_*_${LEN}_${PADDING}.npz -o $OUTFILE --padding 1 -z
#fi
#
#OUTFILE="$BASEDIR/fig_indices_${LEN}_${PADDING}.pdf"
#if [ ! -e $OUTFILE ]
#then
#    python plot_indices.py $BASEDIR/indices_*_${LEN}_${PADDING}.npz -o $OUTFILE --padding 1 -z
#fi

OUTFILE="$BASEDIR/fig_development_${LEN}_${PADDING}.pdf"
python plot_development.py $BASEDIR/synapse_*_${LEN}_${PADDING}.npz -i $BASEDIR/indices_*_${LEN}_${PADDING}.npz -o $OUTFILE --padding 1 -z


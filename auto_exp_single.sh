
trap signalExit 2
function signalExit(){
    echo "\r"
    kill $(jobs -p)
    exit 2
}

if [ $# -lt 4 ]
then
    echo "argument error: usage: bash $0 config_file outdir N length ANALYSIS_ONLY(OPTION)"
    exit 1
fi

CONFIG_FILE=$1
OUTDIR=$2
N=$3
LENGTH=$4

ANALYSIS_ONLY=0
if [ $# -gt 4 ]
then
    ANALYSIS_ONLY=1
fi

if [ ! -d $OUTDIR ]
then
    mkdir $OUTDIR
fi

BLOCKS=(`echo $CONFIG_FILE | tr '/' '\n'`)
CF_COPIED=$OUTDIR/${BLOCKS[${#BLOCKS[@]}-1]}
if [ ! -e $CF_COPIED ]
then
    echo COPY $CONFIG_FILE TO $OUTDIR
    cp $CONFIG_FILE $OUTDIR/
else
    echo $CF_COPIED
fi

if [ $ANALYSIS_ONLY == 0 ]
then
for I in `seq 1 $N`
do
    I_P=`printf %02d $I`
    IS_FILE=`echo $CF_COPIED | \
    sed -e "s/config/states/" | \
    sed -e "s/\.json/_time0000_${I_P}\.npz/"`
    echo $IS_FILE

    if [ ! -e $IS_FILE ]
    then
        python snn_spont.py $CF_COPIED $IS_FILE -l 0 --parallel &
    else
        echo "PASS: $IS_FILE already exists"
    fi
done
wait
for I in `seq 1 $N`
do
    I_P=`printf %02d $I`
    S_FILE=`echo $CF_COPIED | \
    sed -e "s/config/states/" | sed -e "s/\.json/_time0000_${I_P}\.npz/"`
    bash record.sh $CF_COPIED $S_FILE --parallel &
done
wait
TIMES=`seq $LENGTH`
for TIME in ${TIMES[@]}
do
    TIME_P=`printf %04d $TIME`
    TIME_PRE=`expr $TIME - 1`
    TIME_PRE_P=`printf %04d $TIME_PRE`

    for I in `seq 1 $N`
    do
        I_P=`printf %02d $I`
        S_FILE=`echo $CF_COPIED | \
        sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}_${I_P}\.npz/"`
        IS_FILE=`echo $CF_COPIED | \
        sed -e "s/config/states/" | \
        sed -e "s/\.json/_time${TIME_PRE_P}_${I_P}\.npz/"`
        SEED=`expr $I \* $LENGTH + $TIME`

        if [ ! -e $S_FILE ]
        then
            python snn_spont.py $CF_COPIED $S_FILE \
            -s $IS_FILE -l 1 --seed $SEED --parallel &
        else
            echo "PASS: $S_FILE already exists"
        fi
    done
    wait
    for I in `seq 1 $N`
    do
        I_P=`printf %02d $I`
        S_FILE=`echo $CF_COPIED | \
        sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}_${I_P}\.npz/"`
        bash record.sh $CF_COPIED $S_FILE --parallel &
    done
    wait
done
fi # ANALYSIS_ONLY == 0

for I in `seq 1 $N`
do
    I_P=`printf %02d $I`
    S_FILE=`echo $CF_COPIED | \
    sed -e "s/config/states/" | sed -e "s/\.json/_time0000_${I_P}\.npz/"`
    bash plot.sh $CF_COPIED $S_FILE --parallel &
done
wait
TIMES=`seq $LENGTH`
for TIME in ${TIMES[@]}
do
    TIME_P=`printf %04d $TIME`
    TIME_PRE=`expr $TIME - 1`
    TIME_PRE_P=`printf %04d $TIME_PRE`

    for I in `seq 1 $N`
    do
        I_P=`printf %02d $I`
        S_FILE=`echo $CF_COPIED | \
        sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}_${I_P}\.npz/"`
        bash plot.sh $CF_COPIED $S_FILE --parallel &
    done
    wait
done

TIME_P=`printf %04d $LENGTH`
TIMES=`seq 0 $LENGTH`
OUTFILES_S=()
OUTFILES_I=()
for I in `seq 1 $N`
do
    I_P=`printf %02d $I`
    OUTFILE_S=`echo $CF_COPIED | \
    sed -e 's/config/synapse/'`_time${TIME_P}_${I_P}.npz
    OUTFILE_I=`echo $CF_COPIED | \
    sed -e 's/config/indices/'`_time${TIME_P}_${I_P}.npz
    FIGNAME_S=`echo $CF_COPIED | \
    sed -e 's/config/synapse/'`_time${TIME_P}_${I_P}.pdf
    FIGNAME_I=`echo $CF_COPIED | \
    sed -e 's/config/indices/'`_time${TIME_P}_${I_P}.pdf

    OUTFILES_S=("${OUTFILES_S[@]} $OUTFILE_S")
    OUTFILES_I=("${OUTFILES_I[@]} $OUTFILE_I")

    S_FILES=()
    R_FILES=()
    for TIME in ${TIMES[@]}
    do
        TIME_P=`printf %04d $TIME`
        S_FILE=`echo $CF_COPIED | \
        sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}_${I_P}\.npz/"`
        #R_FILE=`echo $S_FILE | sed -e "s/states/record_600/"`
        R_FILE=`echo $S_FILE | sed -e "s/states/record_1800/"`

        S_FILES=("${S_FILES[@]} $S_FILE")
        R_FILES=("${R_FILES[@]} $R_FILE")
    done

    if [ ! -e $OUTFILE_S ]
    then
        python analyze_synapse.py ${S_FILES[@]} -o $OUTFILE_S &
    fi
    if [ ! -e $OUTFILE_I ]
    then
        python analyze_indices.py ${R_FILES[@]} -o $OUTFILE_I &
    fi
    wait

    #if [ ! -e $FIGNAME_S ]
    #then
    #    python plot_development.py $OUTFILE_S -o $FIGNAME_S &
    #fi
    #if [ ! -e $FIGNAME_I ]
    #then
    #    python plot_indices.py $OUTFILE_I -o $FIGNAME_I &
    #fi
    #wait
done

#FIGNAME_S=`echo $CF_COPIED | \
#sed -e 's/config/synapse/'`_time${TIME_P}.pdf
#if [ ! -e $FIGNAME_S ]
#then
#    python plot_development.py ${OUTFILES_S[@]} -o $FIGNAME_S
#fi
#
#FIGNAME_I=`echo $CF_COPIED | \
#sed -e 's/config/indices/'`_time${TIME_P}.pdf
#if [ ! -e $FIGNAME_I ]
#then
#    python plot_indices.py ${OUTFILES_I[@]} -o $FIGNAME_I
#fi

FIGNAME_D=`echo $CF_COPIED | \
sed -e 's/config/development/'`_time${TIME_P}.pdf
python plot_development.py ${OUTFILES_S[@]} -i ${OUTFILES_I[@]} -o $FIGNAME_D


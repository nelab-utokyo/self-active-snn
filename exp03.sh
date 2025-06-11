
trap signalExit 2
function signalExit(){
    echo "\r"
    exit 2
}

if [ $# -ne 3 ]
then
    echo "argument error: usage: bash $0 C_FILE OUTDIR LENGTH"
    exit 1
fi

C_FILE=$1
OUTDIR=$2
LENGTH=$3

TIME_P=`printf %04d $LENGTH`
BLOCKS=(`echo $C_FILE | tr '/' '\n'`)
BLOCKS=(`echo ${BLOCKS[${#BLOCKS[@]}-1]} | tr '.' '\n'`)
OUTFILE_S=$OUTDIR/`echo ${BLOCKS[0]} | sed -e 's/config/synapse/'`_time$TIME_P.npz
OUTFILE_I=$OUTDIR/`echo ${BLOCKS[0]} | sed -e 's/config/indices/'`_time$TIME_P.npz
FIGNAME_S=$OUTDIR/`echo ${BLOCKS[0]} | sed -e 's/config/synapse/'`_time$TIME_P.pdf
FIGNAME_I=$OUTDIR/`echo ${BLOCKS[0]} | sed -e 's/config/indices/'`_time$TIME_P.pdf

S_FILES=()
R_FILES=()
TIMES=`seq 0 $LENGTH`
for TIME in ${TIMES[@]}
do
    TIME_P=`printf %04d $TIME`

    S_FILE=`echo $C_FILE | sed -e "s/config/states/" | sed -e "s/\.json/_time${TIME_P}\.npz/"`
    #R_FILE=`echo $S_FILE | sed -e "s/states/record_600/"`
    R_FILE=`echo $S_FILE | sed -e "s/states/record_1800/"`

    S_FILES=("${S_FILES[@]} $S_FILE")
    R_FILES=("${R_FILES[@]} $R_FILE")
done

# analyze
if [ ! -e $OUTFILE_S ]
then
    python analyze_synapse.py ${S_FILES[@]} -o $OUTFILE_S
fi
if [ ! -e $OUTFILE_I ]
then
    python analyze_indices.py ${R_FILES[@]} -o $OUTFILE_I
fi

# plot
if [ ! -e $FIGNAME_S ]
then
    python plot_development.py $OUTFILE_S -o $FIGNAME_S
fi
# if [ ! -e $FIGNAME_I ]
# then
#     python plot_indices.py $OUTFILE_I -o $FIGNAME_I
# fi

#for S_FILE in ${S_FILES[@]}
#do
#    echo $S_FILE
#done
#
#for R_FILE in ${R_FILES[@]}
#do
#    echo $R_FILE
#done


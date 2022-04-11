set -eu

trap signalExit 2
function signalExit(){
    echo "\r"
    exit 2
}

if [ $# -le 1 ]
then
    echo "argument error: usage: bash $0 C_FILE S_FILE PARALLEL(OPTION)"
    exit 1
fi
OPTION=""
if [ $# -ge 3 ]
then
    OPTION="--parallel"
fi

#RUNTIME_S=2
RUNTIME_L=1800

#RUNTIME_S_P=`printf %03d $RUNTIME_S`
RUNTIME_L_P=`printf %04d $RUNTIME_L`
#RECORD_FILE_S=`echo $2 | sed -e "s/states/record_${RUNTIME_S_P}/"`
RECORD_FILE_L=`echo $2 | sed -e "s/states/record_${RUNTIME_L_P}/"`
FIG_STATE=`echo $2 | sed -e "s/states/fig_state/" | sed -e "s/npz/pdf/"`
FIG_STAT=`echo $2 | sed -e "s/states/fig_stat/" | sed -e "s/npz/pdf/"`
FIG_WEIGHT=`echo $2 | sed -e "s/states/fig_weight/" | sed -e "s/npz/pdf/"`

# PLOT
#if [ ! -e $FIG_STATE ]
#then
#    python plot_state.py $1 $RECORD_FILE_S -o $FIG_STATE
#fi

if [ ! -e $FIG_STAT ]
then
    python plot_stats.py $1 $RECORD_FILE_L -o $FIG_STAT -l 120
fi

if [ ! -e $FIG_WEIGHT ]
then
    python plot_connectivity.py $2 -o $FIG_WEIGHT
fi


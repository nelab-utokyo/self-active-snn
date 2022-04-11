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

## RECORD
#if [ ! -e $RECORD_FILE_S ]
#then
#    python snn_record.py $1 $2 $RECORD_FILE_S -l $RUNTIME_S
#fi

if [ ! -e $RECORD_FILE_L ]
then
    python snn_record.py $1 $2 $RECORD_FILE_L -l $RUNTIME_L --spike_only $OPTION
fi


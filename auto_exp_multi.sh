
trap signalExit 2
function signalExit(){
    echo "\r"
    kill $(jobs -p)
    exit 2
}

if [ $# -ne 3 ]
then
    echo "argument error: usage: bash $0 exp_setting_file outdir length"
    exit 1
fi

if [ ! -e $1 ]
then
    echo "error: $1 does not exist"
    exit 1
elif [ ! -d $2 ]
then
    mkdir $2
fi

BLOCKS=(`echo $1 | tr '/' '\n'`)
if [ ! -e $2/${BLOCKS[${#BLOCKS[@]}-1]} ]
then
    echo COPY $1 TO $2
    cp $1 $2/
fi

CONFIG_FILES=`python auto_config_maker.py $1 $2`
for CONFIG_FILE in ${CONFIG_FILES[@]}
do
    echo $CONFIG_FILE
    bash exp01.sh $CONFIG_FILE PARALLEL &
done
wait

for CONFIG_FILE in ${CONFIG_FILES[@]}
do
    bash exp02.sh $CONFIG_FILE $3 PARALLEL &
done
wait

for CONFIG_FILE in ${CONFIG_FILES[@]}
do
    bash exp03.sh $CONFIG_FILE $2 $3 &
done
wait

TIME_P=`printf %04d $3`
#RECORDS=(`ls $2/record_600_*_time$TIME_P.npz`)
RECORDS=(`ls $2/record_1800_*_time$TIME_P.npz`)
python plot_delta.py ${RECORDS[@]} -o $2/deltap_time$TIME_P.pdf


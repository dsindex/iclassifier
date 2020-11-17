#!/bin/bash

set -o nounset
set -o errexit

VERBOSE_MODE=0

function error_handler()
{
  local STATUS=${1:-1}
  [ ${VERBOSE_MODE} == 0 ] && exit ${STATUS}
  echo "Exits abnormally at line "`caller 0`
  exit ${STATUS}
}
trap "error_handler" ERR

PROGNAME=`basename ${BASH_SOURCE}`
DRY_RUN_MODE=0

function print_usage_and_exit()
{
  set +x
  local STATUS=$1
  echo "Usage: ${PROGNAME} [-v] [-v] [--dry-run] [-h] [--help]"
  echo ""
  echo " Options -"
  echo "  -v                 enables verbose mode 1"
  echo "  -v -v              enables verbose mode 2"
  echo "      --dry-run      show what would have been dumped"
  echo "  -h, --help         shows this help message"
  exit ${STATUS:-0}
}

function debug()
{
  if [ "$VERBOSE_MODE" != 0 ]; then
    echo $@
  fi
}

GETOPT=`getopt -o vh --long dry-run,help -n "${PROGNAME}" -- "$@"`
if [ $? != 0 ] ; then print_usage_and_exit 1; fi

eval set -- "${GETOPT}"

while true
do case "$1" in
     -v)            let VERBOSE_MODE+=1; shift;;
     --dry-run)     DRY_RUN_MODE=1; shift;;
     -h|--help)     print_usage_and_exit 0;;
     --)            shift; break;;
     *) echo "Internal error!"; exit 1;;
   esac
done

if (( VERBOSE_MODE > 1 )); then
  set -x
fi


# template area is ended.
# -----------------------------------------------------------------------------
if [ ${#} != 0 ]; then print_usage_and_exit 1; fi

# current dir of this script
CDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]})))
PDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]}))/..)

# -----------------------------------------------------------------------------
# functions

function make_calmness()
{
    exec 3>&2 # save 2 to 3
    exec 2> /dev/null
}

function revert_calmness()
{
    exec 2>&3 # restore 2 from previous saved 3(originally 2)
}

function close_fd()
{
    exec 3>&-
}

function jumpto
{
    label=$1
    cmd=$(sed -n "/$label:/{:a;n;p;ba};" $0 | grep -v ':$')
    eval "$cmd"
    exit
}


# end functions
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# main 

make_calmness
if (( VERBOSE_MODE > 1 )); then
    revert_calmness
fi

mkdir -p ${CDIR}/wdir
WDIR=${CDIR}/wdir

function prepare_resources_glove {
  local config_file=$1
  local data_dir=$2
  cp -rf ${config_file}               ${WDIR}/mconfig.json
  cp -rf ${data_dir}/label.txt        ${WDIR}/
  cp -rf ${data_dir}/embedding.npy    ${WDIR}/
  cp -rf ${data_dir}/vocab.txt        ${WDIR}/mvocab.txt
}

function prepare_resources_bert {
  local config_file=$1
  local data_dir=$2
  cp -rf ${config_file}                                ${WDIR}/mconfig.json
  cp -rf ${data_dir}/label.txt                         ${WDIR}/
  cp -rf ${PDIR}/bert-checkpoint/config.json           ${WDIR}/
  cp -rf ${PDIR}/bert-checkpoint/tokenizer_config.json ${WDIR}/
  cp -rf ${PDIR}/bert-checkpoint/vocab.txt             ${WDIR}/
}
  
extra_files_glove="${PDIR}/model.py,${PDIR}/tokenizer.py,${PDIR}/util.py,${WDIR}/mconfig.json,${WDIR}/label.txt,${WDIR}/embedding.npy,${WDIR}/mvocab.txt"
extra_files_bert="${PDIR}/model.py,${PDIR}/tokenizer.py,${PDIR}/util.py,${WDIR}/mconfig.json,${WDIR}/label.txt,${WDIR}/config.json,${WDIR}/tokenizer_config.json,${WDIR}/vocab.txt"
function archive {
  local model_name=$1
  local model_file=$2
  local extra_files=$3
  rm -rf *.mar
  torch-model-archiver \
      --model-name ${model_name} \
      --version 1.0 \
      --serialized-file ${model_file} \
      --extra-files ${extra_files} \
      --handler ${CDIR}/handler.py \
      --runtime python3
}

config_file=${PDIR}/configs/config-bert-cls.json
data_dir=${PDIR}/data/clova_sentiments
model_name='electra'
model_file=${PDIR}/pytorch-model.pt
extra_files=${extra_files_bert}

prepare_resources_bert ${config_file} ${data_dir}

archive ${model_name} ${model_file} ${extra_files}

mkdir -p ${CDIR}/model_store
cp -rf ${CDIR}/${model_name}.mar ${CDIR}/model_store

close_fd

# end main
# -----------------------------------------------------------------------------

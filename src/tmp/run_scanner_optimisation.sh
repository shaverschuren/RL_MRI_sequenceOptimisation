#!/usr/bin/env bash

set -Eeuo pipefail

# Setup directories
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
src="$(dirname "$dir")"
root="$(dirname "$src")"

cd $root

usage() {
	cat<<EOF
Usage: $(basename "${BASH_SOURCE[0]}") -m snr|cnr [-h] [-s] [-vm] [-v]

This script runs the entire protocol for a flip angle optimisation
via our reinfocement learning model and MATLAB-based scanner interface.
The first parameter passed should be either snr or cnr and is used to switch
between our two types of optimisers.
The options given below may be used to change this script's behaviour.

Available options:
-m, --mode		(required) snr|cnr - Choose snr/cnr optimisation
-h, --help      	Print this help and exit
-s, --simulation	Use the MATLAB simulator instead of scanner interface
-vm, --simulate_vm  	Simulate the interface VM in Python, only available with -s
-v, --validation	Run the validation program instead of optimisation
EOF
	exit
}

parse_params() {
  	# Default values
  	mode=''
  	simulation=0
  	vm=0
	validation=0

	# Parse passed options
  	while :; do
    	case "${1-}" in
    	-h | --help) usage ;;
    	-s | --simulation) simulation=1 ;;
		-vm | --simulate_vm) vm=1 ;;
		-v | --validation) validation=1 ;;
    	-m | --mode)
      	mode="${2-}"
      	shift
      	;;
    	-?*) die "Unknown option: $1" ;;
    	*) break ;;
    	esac
    	shift
  	done

  	args=("$@")

  	# check required params and arguments
  	[[ -z "${mode-}" ]] && die "Missing required parameter: mode"

	# Check whether mode is either snr or cnr
	if ! [[ "$mode" =~ ^(snr|cnr)$ ]]; then die "'mode' parameter should be either snr or cnr"; fi

  	return 0
}

msg() {
  	echo >&2 -e "${1-}"
}

die() {
  	local msg=$1
  	local code=${2-1} # default exit status 1
  	msg "$msg"

	# Kill all subprocesses
	pkill -P $$
	# Exit
  	exit "$code"
}

# Setup interrupt management
trap 'die "User interruption"' INT

# Setup parameters
parse_params "$@"

# Get conda info and activate proper environment
conda_dir=$(conda info | grep -i 'base environment')
conda_dir=${conda_dir:26}
end_idx=$(expr index "$conda_dir" " ")-1
conda_dir=${conda_dir:0:$end_idx}

source "$conda_dir/etc/profile.d/conda.sh"

conda activate r_learning

# Start up vm simulator (if applicable)
if [ $vm -eq 1 ]
then
	msg "Starting up vm simulator... "
	python "$dir/vm_sim.py" >/dev/null &
fi

# Start up MATLAB interface
if [ $simulation -eq 0 ]
then
	msg "Starting up MATLAB interface..."
	matlab -nodisplay -nosplash -r "run('$src/scanner_interface/rmi_real_time_feedback.m')" >/dev/null &
elif [ $simulation -eq 1 ]
then
	msg "Starting up MATLAB simulator..."
	matlab -nodisplay -nosplash -r "run('$src/scanner_interface/rmi_real_time_feedback_simulator.m')" >/dev/null &
fi

# Run reinforcement learning program (or validation)
if [ $validation -eq 0 ]
then
	msg "Running RL loop..."
	python "$src/scanner_optimizers/"$mode"_optimizer.py"
elif [ $validation -eq 1 ]
then
	msg "Running validation loop..."
	python "$src/validators/scanner_"$mode"_validator.py"
fi

# Exit program
die "Finished!" 0

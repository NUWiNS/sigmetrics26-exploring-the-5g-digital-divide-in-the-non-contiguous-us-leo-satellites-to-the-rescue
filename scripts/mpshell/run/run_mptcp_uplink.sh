#!/bin/bash

sudo sysctl -w net.ipv4.tcp_no_metrics_save=1
sudo sysctl -w net.ipv4.route.flush=1

sudo sysctl -w net.ipv4.tcp_rmem='450000000 450000000 450000000'
sudo sysctl -w net.core.rmem_default=450000000
sudo sysctl -w net.core.rmem_max=450000000

sudo sysctl -w net.ipv4.tcp_wmem='450000000 450000000 450000000'
sudo sysctl -w net.core.wmem_default=450000000
sudo sysctl -w net.core.wmem_max=450000000

sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# Define the root directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASETS_DIR="${MPSHELL_DATASETS_DIR:-$REPO_ROOT/datasets}"
OUTPUT_DIR="${MPSHELL_OUTPUT_DIR:-$SCRIPT_DIR/output/bbr}"

# Read "test_up" value from test_up.txt
TEST_UP_FILE="$SCRIPT_DIR/test_up"
if [[ -f "$TEST_UP_FILE" ]]; then
    TEST_UP_VALUE=$(cat "$TEST_UP_FILE")
else
    echo "ERROR: test_up.txt not found!"
    exit 1
fi

trace_dir_for_location() {
    case "$1" in
        alaska) echo "$DATASETS_DIR/alaska_road_test_202406/emulation/formated_traces/uplink" ;;
        hawaii) echo "$DATASETS_DIR/hawaii_road_test_202408/emulation/formated_traces/uplink" ;;
        *) return 1 ;;
    esac
}

# Define the test locations
#LOCATIONS=("maine")
LOCATIONS=("hawaii" "alaska")

for LOCATION in "${LOCATIONS[@]}"; do
    LOCATION_TRACE_DIR="$(trace_dir_for_location "$LOCATION")"
    if [[ ! -d "$LOCATION_TRACE_DIR" ]]; then
        echo "WARNING: trace directory missing for $LOCATION: $LOCATION_TRACE_DIR"
        continue
    fi

    for SUBFOLDER in "$LOCATION_TRACE_DIR"/*; do
        if [[ -d "$SUBFOLDER" ]]; then
            SUBFOLDER_NAME=$(basename "$SUBFOLDER")

            RTT_FILE="$SUBFOLDER/RTT.txt"
            declare -A DELAY_MAP
            if [[ -f "$RTT_FILE" ]]; then
                while read -r OP DELAY; do
                    DELAY_MAP[$OP]=$DELAY
                done < "$RTT_FILE"
            else
                echo "WARNING: RTT.txt missing in $SUBFOLDER"
                continue
            fi

            # Read run_log.txt for traces
            RUN_LOG_FILE="$SUBFOLDER/run_log.txt"
            if [[ ! -f "$RUN_LOG_FILE" ]]; then
                echo "WARNING: run_log.txt missing in $SUBFOLDER"
                continue
            fi

            while read -r TRACE_ID TRACE_DURATION; do
                echo "Processing trace: $TRACE_ID with duration $TRACE_DURATION seconds"

                CELL_COUNT=0
                for TRACE_FILE in "$SUBFOLDER/$TRACE_ID"*; do
                    if [[ -f "$TRACE_FILE" ]]; then
                        OP=$(basename "$TRACE_FILE" | sed -E 's/^[0-9]+_[0-9]+-([^-_]+).*/\1/')
                        if (( CELL_COUNT == 0 )); then
                            CELL_FILE_1="$TRACE_FILE"
                            OPERATOR_1="$OP"
                            CELL_DELAY_1="${DELAY_MAP[$OPERATOR_1]}"
                            CELL_COUNT=$((CELL_COUNT + 1))
                        elif (( CELL_COUNT == 1 )); then
                            CELL_FILE_2="$TRACE_FILE"
                            OPERATOR_2="$OP"
                            CELL_DELAY_2="${DELAY_MAP[$OPERATOR_2]}"
                            CELL_COUNT=$((CELL_COUNT + 1))
                        fi
                    fi
                done

                if [[ -z "$CELL_FILE_1" || -z "$CELL_FILE_2" ]]; then
                    echo "ERROR: Missing one or both cellular trace files for $TRACE_ID"
                    continue
                fi

                CELL_SIZE_1=$(stat -c %s "$CELL_FILE_1")
                CELL_SIZE_2=$(stat -c %s "$CELL_FILE_2")

                if [[ $CELL_SIZE_1 -gt 100 && $CELL_SIZE_2 -gt 100 ]]; then

		        # Create output directory for the trace
		        TRACE_OUTPUT_DIR="$OUTPUT_DIR/$LOCATION/$SUBFOLDER_NAME/$TRACE_ID"
		        mkdir -p "$TRACE_OUTPUT_DIR"

		        ## **Run Test**
		        bash -c "
		            echo 'Running test $TRACE_ID for $OPERATOR_1 on RTT $CELL_DELAY_1 and $OPERATOR_2 on RTT $CELL_DELAY_2'
		            mpshell '$CELL_DELAY_1' '$TEST_UP_FILE' '$CELL_FILE_1' '$CELL_DELAY_2' '$TEST_UP_FILE' '$CELL_FILE_2' /usr/bin/stdbuf -oL iperf3 -m -s -i 0.5 | tee '$TRACE_OUTPUT_DIR/mptcp_up_default.txt' & 
		            sleep 1
		            iperf3 -m -c 100.64.0.2 -i 0.5 -t '$TRACE_DURATION' > /dev/null 
		            sleep 2
		            sudo pkill -f iperf3
		            sudo pkill -f mpshell
		            wait
		            sleep 1
		        "
		        
	    		sudo pkill -9 -f iperf3
	    		sleep 1
	    		tput reset
	    	fi

            done < "$RUN_LOG_FILE"
        fi
    done
done

reset

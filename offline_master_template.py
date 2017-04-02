# Import all external dependencies
from psychopy import visual, core, gui
import pickle
import pylsl
import numpy as np

# Import custom code
from custom_module import *
# ^ CUSTOM MODULE IS WHATEVER YOU NAME THE 'BODY' SCRIPT. UPDATE NAME HERE

"""
THIS IS A TEMPLATE FOR AN OFFLINE EXPERIMET.
PARTS IN ALL CAPS ARE NOTES ON THE TEMPLATE, AND NEED UPDATING TO RUN.

This is the master script to run the .... experiment.

Notes:
    The code to run this experiment is in the custom .... module.
    The presentation is all done with psychopy
        http://www.psychopy.org
    Event markers and collecting data together is done with lab-streaming layer (LSL)
        https://github.com/sccn/labstreaminglayer
    The default settings for the experiment are loaded with the ExpInfo() Class.
        These settings can be over-written below for test runs / alternate run set-ups.
    This experiment saves out data in multiple ways:
        - Markers are sent to LSL, to be saved out with the EEG Data
        - CSV files get trial info written out (written out after each trial)
        - At the end of the experiment, experiment data is saved out (as npz)
"""

# Set whether to run a test run or not
TEST_RUN = False

# Set window size & color
win_size = [800, 600]
win_color = [-1, -1, -1]

# Initialize object variables for experiment info, run info and stimuli
exinfo = ExpInfo()
run = RunInfo()
stim = Stim()

## SETTINGS FOR TEST RUN
# If it's a test run, these lines will over ride defaults and run quick test version
# UPDATE THESE PARAMS AS NEEDED FOR EXPERIMENT
if TEST_RUN:
    exinfo.ntrain_1 = 2
    exinfo.ntrain_2 = 2
    exinfo.nthresh_trials = 5
    exinfo.nreversals = 1
    exinfo.step_sizes = [0.1]
    exinfo.nblocks = 1
    exinfo.ntrials_per_block = 6

# Set up Marker Stream (Outgoing Stream)
marker_info = pylsl.stream_info("Exp-Events", "Markers", 1, 0, pylsl.cf_string)
marker_outlet = pylsl.stream_outlet(marker_info)

# Get subject information from GUI - ADD ANYTHING ELSE TO BE COLLECTED AT INITIATION HERE
start_vars = {'Subject Number':''}
start_gui = gui.DlgFromDict(start_vars, order=['Subject Number'])
exinfo.subnum = int(start_vars['Subject Number'])

# Set up files and clock for the experiment
run.make_files(exinfo.subnum)
run.make_clock()

# Create window - MIGHT NEED TO UPDATE MONITOR SIZE
mywin = visual.Window(win_size, color=win_color, colorSpace='rgb', fullscr=True,
                      monitor="MON_NAME", units="norm")
mywin.setRecordFrameIntervals(True)

# Check the monitor
exinfo = check_monitor(mywin, exinfo)

# Make the stimuli
stim.make_stim(mywin)

# Write Experiment logfile
experiment_log(exinfo, run, stim)

# Run training
print "Training"
train_exp_data = train(mywin, EEGinlet, marker_outlet, exinfo, run, stim)

# Run Thresholding
print "Thresholding"
thresh_exp_data, stim = threshold_staircase(mywin, EEGinlet, marker_outlet, exinfo, run, stim)

# Run Experiment Blocks
print "Blocks"
for bl in range(exinfo.nblocks):
    bl_exp_data, bl_meth_data = run_block(mywin, EEGinlet, marker_outlet, exinfo, run, stim)
    exinfo.update_block_number()
    if bl == 0:
        exp_data = bl_exp_data
        method_data = bl_meth_data
    else:
        exp_data = np.vstack([exp_data, bl_exp_data])
        method_data = np.vstack([method_data, bl_meth_data])

# Save data to npz file
npz_path = "PATH_TO_SAVE_FILE_TO"
npz_save_name = npz_path + str(exinfo.subnum) + '_' + exinfo.datetimenow
np.savez(npz_save_name, subnum=exinfo.subnum, train_exp_data=train_exp_data,
         thresh_exp_data=thresh_exp_data, exp_data=exp_data, )

# Save experiment objects using pickle
pickle_path = "PATH_TO_SAVE_FILE_TO"
pickle.dump(exinfo, open((pickle_path + str(exinfo.subnum) + '_exinfo.p'), 'wb'))
pickle.dump(run, open((pickle_path + str(exinfo.subnum) + '_run.p'), 'wb'))
pickle.dump(stim, open((pickle_path + str(exinfo.subnum) + '_stim.p'), 'wb'))

# Save Frame Intervals
mywin.saveFrameIntervals(fileName="PATH_TO_SAVE_FILE_TO", clear=True)

# End experiment
mywin.close()

core.wait(20.0)

core.quit()

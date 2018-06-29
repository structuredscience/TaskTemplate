"""
THIS IS A TEMPLATE FOR A ONLINE EXPERIMET - USING REAL-TIME PHASE PREDICTION.
PARTS IN ALL CAPS ARE NOTES ON THE TEMPLATE, AND NEED UPDATING TO RUN.

Classes & Functions to run the .... experiment.

Notes:
    - This script has support for re-referencing online data collection, but it
        is not used in the rtPB-3 protocol.
    - Here, set up to use LSL for sending event markers. This can be changed.
    - By default, this template will collect and save out data on the online methods.
"""

# Import all external dependencies
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

import pylsl
from psychopy import visual, core, gui, data, event

# Import the real-time functions
from RTPP import *

###################################################################################################
#################################### rt exp template - Classes ####################################
###################################################################################################

class ExpInfo(object):
    """Class to store experiment run parameters. """

    def __init__(self):

        # Initialize subject number field
        self.subnum = int()

        # Experiment version
        self.runversion = ''
        self.dateversion = ''
        self.eegsystem = ''

        # Run time
        self.datetimenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Experiment Block Settings
        self.nblocks = int()
        self.block_number = 1
        self.ntrials_per_block = int()

        # Thresholding Settings - FOR BASIC STAIRCASE. CHANGE FOR OTHER/NO THRESHOLDING
        self.nthresh_trials = int()
        self.nreversals = int()
        self.step_sizes = []# List of ints/floats

        # Label Block Settings - IF DOING LABEL BLOCKS, SETTINGS HERE
        self.nlabel_blocks = int()
        self.ntrials_label_block = int()

        # Setting for rest-EEG collection (psdcheck)
        self.rest_length = float()      # Time, in seconds, to collect rest EEG

        # Settings for train
        self.ntrain_1 = int()      # Number of trials in first set of practice trials
        self.ntrain_2 = int()      # Number of trials in second set of practice trials

        # Check current lsl version
        self.lsl_protocol_version = pylsl.protocol_version()
        self.lsl_library_version = pylsl.library_version()

        # Set booleans for what have been checked
        self.check_stream = False
        self.check_monitor = False


    def update_block_number(self):
        """Increment block number after running a block. """
        self.block_number += 1


class RunInfo(object):
    """Class to store information details used to run individual trials. """

    def __init__(self):

        # Trial stimuli settings - ADD EXPERIMENT SPECIFIC VARIABLES HERE
        self.wait_time = float()        # Time to wait for a response (in seconds)
        self.iti = float()              # Time to wait between trials (in seconds)

        # EEG settings
        self.reref = False          # Whether to re-reference online
        self.chans = []             # EEG channel(s) to use. If re-ref: 2nd item is reference.
        self.srate = float()        # Sampling rate of the incoming EEG data
        self.nsamples = int()       # Number of samples to use for analysis & prediction

        # Frequency and filter initializations
        self.center_freq = float()
        self.bp = np.array([])

        # Add a dictionary of event marker names for detection trial types
        self.ev_names = dict({0: 'Peak',
                              0.5: 'Fall',
                              1: 'Trough',
                              1.5: 'Rise',
                              2: 'Undetected',
                              3: 'Sham'})

        # Initialize file name vars
        self.dat_fn = None
        self.dat_f = None
        self.meth_fn = None
        self.meth_f = None

        # Initialize clock var
        self.clock = None


    def make_files(self, subnum):
        """Initialize data files and write headers. """

        # Experiment Data File
        self.dat_fn = "PATH_TO_SAVE_FILE_TO" + str(subnum) + "_exp.csv"
        self.dat_f = open(self.dat_fn, 'w')
        self.dat_f.write('PUT HEADERS HERE')
        self.dat_f.close()

        # Methods Data File
        self.meth_fn = "PATH_TO_SAVE_FILE_TO" + str(subnum) + "_method.csv"
        self.meth_f = open(self.meth_fn, 'w')
        self.meth_f.write('PUT HEADERS HERE')
        self.meth_f.close()


    def make_clock(self):
        """Make the clock to use for the experiment. """
        self.clock = core.Clock()


class Stim(object):
    """Class to store all the stimuli used in the experiment. """

    def __init__(self):

        # Boolean for whether stimuli have been created
        self.stim = False

        # ADD ALL PARAMETERS TO CREATE STIMULI HERE
        # DON'T ACTUALLY CREATE THE STIM AT INIT
        self.fix_height = int()  # <- EXAMPLE PARAM

        # INITIALIZE VARS THAT WILL STORE STIM
        self.fix = None


    def make_stim(self, mywin):
        """Create the stimuli for the experiment."""

        # Set boolean that stim have been created
        self.stim = True

        # Fixation Cross - KEEP IF YOU NEED A FIXATION CROSS
        self.fix = visual.TextStim(mywin, '+', height=self.fix_height, pos=[0, 0])

        # CREATE OTHER STIMULI


    def update_stim(self, new_stim_param):
        """Update luminance values of flash. Used in behavioural thresholding.

        Parameters
        ----------
        self : Stim() object
            Object of all stimuli for the task.
        new_stim_param : float
            New value to update stimuli to.

        NOTE: MIGHT HAVE OTHER INPUTS, SUCH AS WHICH SIDE YOU ARE UPDATING.
        """

        # UPDATE STIMULI
        pass


class Block(object):
    """Class to store information to run a block of trials."""

    def __init__(self):

        # Initiliaze arrays to store trial parameters for a block
        # INITIALIZE ARRAYS FOR TRIAL PARAMETERS HERE
        #  FOR EXAMPLE, TRIAL CONDITION, SIDE OF PRESENTATION, ETC.
        pass


class Inds(object):
    """Class to store index number for data storage.

    Notes:
    - An index object is used to be able to index data with
        meaningful variables, instead of 'magic' numbers.
    """

    def __init__(self):

        # Index numbers in trial output data vector
        self.block = 0
        self.trial = 1
        # ADD MORE INDICES HERE


###################################################################################################
################################### rt exp template - Functions ###################################
###################################################################################################

def run_block(mywin, EEGinlet, marker_outlet, exinfo, run, stim):
    """Runs a blocks of trials in the experiment.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    marker_outlet : pylsl StreamOutlet() object
        LSL output stream to send event markers.
    exinfo : ExpInfo() object
        Object of information about the experiment.
    run : RunInfo() object
        Object of information / parameters to run task trials.
    stim : Stim() object
        Object of all stimuli for the task.

    Returns
    -------
    exp_block_data : 2d array
        Matrix of experiment data from the block.
    meth_block_data : 2d array
        Matrix of method data from the block.
    """

    # Get block settings
    ntrials_block = exinfo.ntrials_per_block
    block_num = exinfo.block_number

    # Set up matrix for block data
    exp_block_data = np.zeros(shape=(ntrials_block, LEN_EXP_TRIAL_DAT_OUT))
    meth_block_data = np.zeros(shape=(ntrials_block, LEN_METH_TRIAL_DAT_OUT))

    # Get block trial run info
    block, order = make_block_trials(ntrials_block)
    # ^ SETTINGS FROM HERE SHOULD BE PUT INTO TRIAL_INFO TO SET TRIAL PARAMETERS

    # Send marker for start of block
    marker_outlet.push_sample(pylsl.vectorstr(["Start Block"]))

    # Beginning of block message
    message = visual.TextStim(mywin, text='')
    disp_text(mywin, message, 'Press space when you are ready to start a block of trials.')

    # Pause before block starts, then start with a fixation cross
    core.wait(1.0)
    stim.fix.draw()
    core.wait(1.5)

    # Loop through trials
    for trial in range(0, ntrials_block):
        trial_info = [block_num, ETC] # <- LIST OF INFORMATION TO RUN THE TRIAL
                        #  FOR EXAMPLE: [BLOCK_NUM, TRIAL_NUM, trial_type, SIDE, ETC.]
        trial_dat, method_dat = run_trial_online(
            mywin, EEGinlet, marker_outlet, run, stim, trial_info)
        exp_block_data[trial, :] = trial_dat
        meth_block_data[trial, :] = method_dat
        iti = run.iti + random.random()
        core.wait(iti)

    # Send marker for end of block
    marker_outlet.push_sample(pylsl.vectorstr(["End Block"]))

    # End of block message
    disp_text(mywin, message, 'That is the end of this block of trials.')

    # Return block data
    return exp_block_data, meth_block_data


def make_block_trials(ntrials_block):
    """Creates a matrix of pseudo-random balanced trial parameters for a block of trials.

    Parameters
    ----------
    ntrials_block : int
        Number of trials in the block

    Returns
    -------
    block : 2d array
        Matrix of trial parameters (this is NOT random).
    order : 1d array
        Randomized order to run the trials in.
    """

    ## CREATE VECTORS OF TRIAL PARAMETER SETTINGS FOR A BLOCK OF TRIALS
    # FOR EXAMPLE: COND_VEC = NP.APPEND(NP.ZEROS(NTRIAL_BLOCK/2), NP.ONES(NTRIAL_BLOCK/2))
    #  ^ CREATES A VECTOR TO HAVE 50% OF EACH OF TWO TRIAL CONDITIONS

    # Collect run details into block object
    block = Block()
    # ADD BLOCK RUN
    #  EXAMPLE: block.CONDITION = COND_VEC

    # Set up array for run order
    order = range(0, len(ntrials_block))
    random.shuffle(order)

    return block, order


def run_trial_online(mywin, EEGinlet, marker_outlet, run, stim, trial_info):
    """This function runs the online experiment trials (including sham).

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    marker_outlet : pylsl StreamOutlet() object
        LSL output stream to send event markers.
    run : RunInfo() object
        Object of information / parameters to run task trials.
    stim : Stim() object
        Object of all stimuli for the task.
    trial_info : 1d array
        Vector of trial parameters.
            trial_info: [] <- LIST OF TRIAL INPUT PARAMETERS

    Returns
    -------
    trial_dat : 1d array
        Vector of trial data about behaviour.
            trial_dat: [] <- VECTOR OF TRIAL DATA. SET UP FOR WHAT YOU WANT TO COLLECT.
    method_dat : 1d array
        Vector of trial data about the online method.
            method_dat: [block_#, trial_#, trial_type, time_pres, method[0], method[1], method[2]]
                        method (thresh):  [run_time, top_percentile, bot_percentile]
                        method (filt):    [run_time, peak_width, trough_width]
                - run_time is in ms
                - *_width is in samples (with native srate of 5000.)
    """

    # Get index object
    i = Inds()

    # Get trial paramters
    # PULL OUT TRIAL SETTINGS FROM TRIAL_INFO
    #  FOR EXAMPLE, trial_type_str, SIDE etc.

    # Get fixation cross
    fix = stim.fix

    # Set up trial, pull out stimuli to use for the trial
    ## USE INPUT TRIAL PARAMETERS TO PULL OUT REQUIRED STIM
    trial_type = trial_info[TYPE_IND]
    # ^ FOR EXAMPLE. ADD AS NEEDED.

    # Run the trial
    fix.draw()
    # DRAW ANY TRIAL MARKERS
    mywin.flip()
    marker_outlet.push_sample(pylsl.vectorstr(["Markers"]))
    core.wait(WAIT_TIME + random.random()/10.)

    ## THE FOLLOWING ASSUMES ASSUMES A 'TRIAL_TYPE' PARAMETER, WITH THE FOLLOWING:
    ##   0: train trial; 1: threshold trial; 2: sham online exp trial
    ##   3: filt online exp trial; 4: thresh online exp trial
    ## ^ UPDATE THIS AS REQUIRED

    ## Pull EEG Data using specified method
    # Randomly use either method if train, thresh or sham trial
    if trial_type == 0 or trial_type == 1 or trial_type == 2:
        if trial_type == 0:
            trial_type_str = 'train'
        elif trial_type == 1:
            trial_type_str = 'threshold'
        elif trial_type == 2:
            trial_type_str = 'sham'

        # Randomly pull from either thresh or filt method
        if int(round(random.random())) == 0:
            det_type, pred_prestime, method_data = sample_data_filt(
                EEGinlet, run.chans, run.nsamples, run.srate, run.reref,
                run.bp, run.clock)
            time_pres = pred_prestime - (run.clock.getTime() * 1000)
        else:
            det_type, pred_prestime, method_data = sample_data_thresh(
                EEGinlet, run.chans, run.nsamples, run.srate, run.reref,
                run.center_freq, run.clock)
            time_pres = pred_prestime - (run.clock.getTime() * 1000)
        # Randomize presentation time
        det_type = 3
        pred_prestime = (run.clock.getTime() * 1000) + (random.random() * 100)

    # Filt method trials
    elif trial_type == 3:
        trial_type_str = 'filt'
        det_type, pred_prestime, method_data = sample_data_filt(
            EEGinlet, run.chans, run.nsamples, run.srate, run.reref,
            run.bp, run.clock)
        time_pres = pred_prestime - (run.clock.getTime() * 1000)

    # Thresh method trials
    elif trial_type == 4:
        trial_type_str = 'thresh'
        det_type, pred_prestime, method_data = sample_data_thresh(
            EEGinlet, run.chans, run.nsamples, run.srate, run.reref,
            run.center_freq, run.clock)
        time_pres = pred_prestime - (run.clock.getTime() * 1000)

    # Get current time
    now = run.clock.getTime()*1000

    # Set up event marker
    event_out = run.ev_names[det_type] + '_Pres_' + str(trial_type_str)

    # Present stimulus at predicted time
    while True:
        if now < pred_prestime:
            now = run.clock.getTime()*1000
        else:
            marker_outlet.push_sample(pylsl.vectorstr([event_out]))
            for frame in range(run.flash_nflips):
                fix.draw()
                # DRAW PHASE LOCKED PRESENTATION ITEMS
                mywin.flip()
            break

    # Check time at moment of presentation (offset)
    pres_time = run.clock.getTime()       # In seconds

    # Draw any markers and fixation after the presentation
    fix.draw()
    # DRAW ANY POST TRIAL MARKERS
    mywin.flip()

    # Wait for response
    resp = event.waitKeys(maxWait=run.wait_time, timeStamped=run.clock)

    # After the wait period, draw only fixation
    fix.draw()
    mywin.flip()

    # Check if detected trial
    if resp is None:
        saw = 0
        rt = None
        marker_outlet.push_sample(pylsl.vectorstr(["MissTrial"]))
    else:
        saw = 1
        rt = resp[0][1] - pres_time     # Time stamp
        marker_outlet.push_sample(pylsl.vectorstr(["HitTrial"]))

    ## Output data
    # Trial Data
    trial_dat = [] # <- SET LIST OF DATA YOU WANT TO COLLECT / RETURN
    # Method Data
    method_dat = [trial_info[i.block], trial_info[i.trial], trial_type, det_type,
                  time_pres, method_data[0], method_data[1], method_data[2]]

    ## Print data to files
    # Experiment Data
    strlist_exp = [] # <- MAKE A LIST OF STRINGS OF SAME DATA FOR FILE
    strlist_exp = ",".join(strlist_exp)
    dat_f = open(run.dat_fn, 'a')
    dat_f.write(strlist_exp + '\n')
    dat_f.close()

    # Method Data
    strlist_meth = [str(trial_info[i.block]), str(trial_info[i.trial]), str(trial_type),
                    str(det_type), str(time_pres), str(method_data[0]), str(method_data[1]),
                    str(method_data[2])]
    strlist_meth = ",".join(strlist_meth)
    meth_f = open(run.meth_fn, 'a')
    meth_f.write(strlist_meth + '\n')
    meth_f.close()

    return trial_dat, method_dat


def run_label_block(mywin, EEGinlet, marker_outlet, exinfo, run, trial_type_str):
    """Runs the online labelling, without any task involved.
    It can be used for labelling data with resting data, eyes open or closed.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    marker_outlet : pylsl StreamOutlet() object
        LSL output stream to send event markers.
    exinfo : ExpInfo() object
        Object of information about the experiment.
    run : RunInfo() object
        Object of information / parameters to run task trials.
    trial_type_str : str
        Method of online prediction to use.
            Options: 'filt', 'thresh'
    """

    # Set up variables
    ntrials = exinfo.ntrials_label_block

    # Display Instructions
    message = visual.TextStim(mywin, text='')
    str_message = ("In this block, you just need to sit still with your eyes closed. "
                   "Press Space when you are ready to begin.")
    disp_text(mywin, message, str_message)
    core.wait(5.0)

    # Check type of block ('filt' or 'thresh')
    if trial_type_str == 'filt':
        marker_outlet.push_sample(pylsl.vectorstr(["Filt Labelling"]))
    elif trial_type_str == 'thresh':
        marker_outlet.push_sample(pylsl.vectorstr(["Thresh Labelling"]))

    ## Run a block of EEG labelling
    # Send start of block label
    marker_outlet.push_sample(pylsl.vectorstr(["Start Labelling Block"]))

    # Loop through trials
    for trial in range(0, ntrials):
        run_label_trial(EEGinlet, marker_outlet, run, trial_type_str)
        iti = 0.5 + random.random()
        core.wait(iti)

    # Send marker for end of block
    marker_outlet.push_sample(pylsl.vectorstr(["End Labelling Block"]))

    # Self-paced break for participants between blocks
    disp_text(mywin, message, 'You have finished this block.')


def run_label_trial(EEGinlet, marker_outlet, run, trial_type_str):
    """Labels peaks & troughs without invoking the task.
    Can be run on subject with their eyes closed.

    Parameters
    ----------
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    marker_outlet : pylsl StreamOutlet() object
        LSL output stream to send event markers.
    run : RunInfo() object
        Object of information / parameters to run task trials.
    trial_type_str : str
        Method of online prediction to use.
            Options: 'filt', 'thresh'
    """

    # Pull EEG data and predict using the filter method
    if trial_type_str == 'filt':
        det_type, predicted_prestime, method_data = sample_data_filt(
            EEGinlet, run.chans, run.nsamples, run.srate, run.reref,
            run.bp, run.clock)
    elif trial_type_str == 'thresh':
        det_type, predicted_prestime, method_data = sample_data_thresh(
            EEGinlet, run.chans, run.nsamples, run.srate, run.reref,
            run.center_freq, run.clock)

    ## Write a label at the time of the peak or trough prediction
    # Get current time
    now = run.clock.getTime()*1000

    # Set up event marker
    label_event_out = 'Label_' + run.ev_names[det_type] + '_' + str(trial_type_str)

    # Loop through, waiting for the time to present
    while True:
        if now < predicted_prestime:
            now = run.clock.getTime()*1000
        else:
            marker_outlet.push_sample(pylsl.vectorstr([label_event_out]))
            break


def psdcheck(EEGinlet, marker_outlet, exinfo, run):
    """Takes in rest EEG data to return individualized alpha frequency.

    Parameters
    ----------
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    exinfo : ExpInfo() object
        Object of information about the experiment.
    run : RunInfo() object
        Object of information / parameters to run behavioural trials.

    Returns
    -------
    alpha : float
        Frequency of the participants' alpha.
    """

    # Initialize
    EEGsample = pylsl.vectorf()
    length_rest = exinfo.rest_length
    nsamples_rest = int(run.srate * length_rest)
    EEG_window_rest = np.zeros(nsamples_rest)

    # Print instructions to screen
    message = visual.TextStim(mywin, text='')
    str_message = ("In this block you need to sit still for about a minute, "
                   "with your eyes closed.")
    disp_text(mywin, message, str_message)
    str_message = "Press space when you are ready, and then close your eyes."
    disp_text(mywin, message, str_message)

    # Pause, then start collection
    print('Starting rest EEG Collection')
    core.wait(2.0)

    # Take time to check how long collection is
    time_before_rest = run.clock.getTime()

    # Clear buffer
    while EEGinlet.pull_sample(EEGsample, 0.0): # Clear buffer
        pass

    ## Collect data
    # Send event code for start of rest-record
    marker_outlet.push_sample(pylsl.vectorstr(["StartRest"]))
    # Re-referenced data
    if run.reref:
        for sample_index in range(0, nsamples_rest):
            EEGinlet.pull_sample(EEGsample)
            EEG_window_rest[sample_index] = EEGsample[run.chans[0]] - EEGsample[run.chans[1]]
    # Non-rereferenced data
    else:
        for sample_index in range(0, nsamples_rest):
            EEGinlet.pull_sample(EEGsample)
            EEG_window_rest[sample_index] = EEGsample[run.chans[0]]

    # Send event code for end of rest-record
    marker_outlet.push_sample(pylsl.vectorstr(["EndRest"]))

    # Get time after, and check how long colleciton is
    time_after_rest = run.clock.getTime()
    rest_collection_time = round(time_after_rest - time_before_rest)

    # Remove DC offset
    EEG_window_rest = EEG_window_rest - np.mean(EEG_window_rest)

    ## Take psd, plot and ask experimenter to check
    freqs, psd = sig.welch(EEG_window_rest, fs=run.srate, window='hanning',
                           nperseg=2*run.srate, noverlap=run.srate/2, nfft=None,
                           detrend='linear', return_onesided=True, scaling='density')

    # Set psd range for visualization
    fwin = [1, 50]
    # Find indices of f values closest to range of interest
    fwin = [np.argmin(abs(freqs-fwin[0])), np.argmin(abs(freqs-fwin[1]))]
    freqs = freqs[fwin[0]:fwin[1]]
    psd = psd[fwin[0]:fwin[1]]

    # Plot psd
    plt.figure(1)
    plt.plot(f, np.log10(psd))
    plt.show()

    # Get start variables from GUI
    alpha_vars = {'Alpha Peak':''}
    start_gui = gui.DlgFromDict(alpha_vars, order=['Alpha Peak'])
    alpha = float(alpha_vars['Alpha Peak'])

    # Return alpha value
    return alpha


def train(mywin, EEGinlet, marker_outlet, exinfo, run, stim):
    """Trains the participant on the task.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    marker_outlet : pylsl StreamOutlet() object
        LSL output stream to send event markers.
    exinfo : ExpInfo() object
        Object of information about the experiment.
    run : RunInfo() object
        Object of information / parameters to run task trials.
    stim : Stim() object
        Object of all stimuli for the task.

    Returns
    -------
    train_exp_data : 2d array
        Matrix of trial data from train trials.
    train_method_data : 2d array
        Matrix of method data from train trials.
    """

    # Get index object
    ind = Inds()

    # Get number of practice trials per practice block
    n_prac_1 = exinfo.ntrain_1
    n_prac_2 = exinfo.ntrain_2

    # Set up trial parameters
    block_num = -1
    trial_num = 0
    trial_info = [block_num, trial_num, ETC] # <- TRIAL RUN SETTINGS FOR PRACTICE TRIALS

    # Set up and display text to explain the task
    message = visual.TextStim(mywin, text='')
    disp_text(mywin, message, "INSTRUCTIONS")

    # Run a Test Trial to show an example
    disp_text(mywin, message, "Lets try a test trial.")
    initial_trial_exp, initial_trial_method = run_trial(
        mywin, marker_outlet, run, stim, trial_info)
    core.wait(1.0)

    # Initialize matrices for train data (train block 1)
    train_exp_1 = np.zeros(shape=(n_prac_1, LEN_EXP_TRIAL_DAT_OUT))
    train_meth_1 = np.zeros(shape=(n_prac_1, LEN_METHOD_TRIAL_DAT_OUT))

    # Run practice block of easy trials
    disp_text(mywin, message, "Lets try a few more practice trials.")

    for trial in range(0, n_prac_1):

        # Run trial
        train_exp_1[trial, :], train_meth_1[trial, :] = run_trial_online(
            mywin, EEGinlet, marker_outlet, run, stim, trial_info)
        core.wait(1.0)

    # Present more instructions IF NEEDED
    disp_text(mywin, message, "MORE INSTRUCTIONS")

    # Initialize matrices for train data (train block 2)
    train_exp_2 = np.zeros(shape=(n_prac_2, LEN_EXP_TRIAL_DAT_OUT))
    train_meth_2 = np.zeros(shape=(n_prac_2, LEN_METHOD_TRIAL_DAT_OUT))

    # Run another practice block of trials
    for trial in range(0, n_prac_2):

        # Run trial
        train_exp_2[trial, :], train_meth_2[trial, :] = run_trial_online(
            mywin, EEGinlet, marker_outlet, run, stim, trial_info)
        core.wait(1.0)

    # Collect all train data together
    train_exp_data = np.vstack([initial_trial_exp, train_exp_1, train_exp_2])
    train_method_data = np.vstack([initial_trial_method, train_meth_1, train_meth_2])

    # Pause to ask the subject if they have any questions about the task
    str_message = ("If you have any questions about the task, "
                   "please ask the experimenter now.")
    disp_text(mywin, message, str_message)

    return train_exp_data, train_method_data


def disp_text(mywin, message, text):
    """Displays text on screen, waits for a keypress, then clears text and continues.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
    message : psychopy visual.TextStim() object
        Psychopy text stimulus object.
    text : str
        Words to display (string).
    """

    # Set the given text in text object
    message.setText(text)

    # Display text, flip screen and wait for key press
    message.draw()
    mywin.flip()
    _ = event.waitKeys()
    mywin.flip()

    return


def threshold_staircase(mywin, EEGinlet, marker_outlet, exinfo, run, stim):
    """Run a psychophysical staircase to set luminance values.
    Here - looking to set parameters for 50 percent detection rate.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    marker_outlet : pylsl StreamOutlet() object
        LSL output stream to send event markers.
    exinfo : ExpInfo() object
        Object of information about the experiment.
    run : RunInfo() object
        Object of information / parameters to run task trials.
    stim : Stim() object
        Object of all stimuli for the task.

    Returns
    -------
    thresh_exp_data : 2d array
        Experiment data from thresholding trials.
    thresh_method_data : 2d array
        Method data from thresholding trials.
    stim : Stim() object
        Stim object updated with thresholded lums.
    """

    # Create the Staircase Handler
    #staircase_left = data.StairHandler(startVal=0.5, nReversals=3, stepSizes=0.01, nTrials=50,
    #    nUp=1, nDown=2, stepType='lin', minVal=0, maxVal=1)

    #staircase_right = data.StairHandler(startVal=0.5, nReversals=3, stepSizes=0.01, nTrials=50,
    #    nUp=1, nDown=2, stepType='lin', minVal=0, maxVal=1)

    # Get index object
    i = Inds()

    # Set up Trial Info
    block_num = 0
    trial_num = 1
    trial_info = [block_num, trial_num, ETC]

    # Set up staircase condition parameters
    stair_conds = [] # <- LIST OF DICTIONARIES TO INITIALIZE STAIRCASES

    # Set up staircase handler
    staircases = data.MultiStairHandler(
        stairType='simple', conditions=stair_conds, nTrials=exinfo.nthresh_trials)

    # Initialize matrices to store thresholding data
    thresh_exp_data = np.zeros([0, LEN_EXP_TRIAL_DAT_OUT])
    thresh_method_data = np.zeros([0, LEN_METHOD_TRIAL_DAT_OUT])

    # Initialize message
    message = visual.TextStim(mywin, text='')
    break_message = ("You may take a short break. "
                     "Press when you are ready for a new block of trials.")

    # Pause before block starts, then start with a fixation cross
    core.wait(1.0)
    stim.fix.draw()
    core.wait(1.5)

    # Run threshold trial
    for thresh_param, cond in staircases:

        # SET UP TRIAL
        #  MIGHT NEED TO EXTRACT CONDITION, ETC.

        # Update stimulus luminance
        stim.update_lum(THRESH_PARAM)

        # Run trial
        exp_trial_dat, meth_trial_dat = run_trial_online(
            mywin, EEGinlet, marker_outlet, run, stim, trial_info)

        # Inter-trial interval
        core.wait(run.iti + random.random())

        # Add subject response to staircase
        staircases.addResponse(exp_trial_dat[RESP_INDEX])

        # Add data to thresh data matrices
        thresh_exp_data = np.vstack([thresh_exp_data, exp_trial_dat])
        thresh_method_data = np.vstack([thresh_method_data, meth_trial_dat])

        # Increment trial number
        trial_num += 1

        # Take a break after a certain number of trials
        if trial_num % 48 == 0:
            disp_text(mywin, message, break_message)

    # Pull out results from staircases
    #  DEPENDING HOW YOU ARE USING STAIRCASES, THIS MIGHT REQUIRED AVERAGING OVER
    #    MULTIPLE STAIRCASES, AND/OR A SET OF REVERSAL POINTS IN EACH STAIRCASE
    UPDATE_PARAMS = None

    # Update stim lums to be used for main experiment
    stim.update_stim(UPDATE_PARAMS)

    # Save staircase object
    staircases.saveAsPickle("PATH_TO_SAVE_FILE_TO" + str(exinfo.subnum) + "_thresh")

    return thresh_exp_data, thresh_method_data, stim


def check_stream(EEGinlet, exinfo):
    """Runs the checks on the current stream. Saves info to ExpInfo() object.

    Parameters
    ----------
    EEGinlet : pylsl StreamInlet() object
        Data stream from LSL for incoming EEG data.
    exinfo : ExpInfo() object
        Object of information about the experiment.

    Returns
    -------
    exinfo : ExpInfo() object
        Object of information about the experiment.
    """

    exinfo.check_stream = True

    # Get Stream Info
    stream_info = EEGinlet.info()

    # Check stream name
    exinfo.stream_name = stream_info.name()

    # Check content type
    exinfo.stream_type = stream_info.type()

    # Check sampling rate
    exinfo.stream_srate = stream_info.nominal_srate()

    # Check number of channels
    exinfo.stream_nchans = stream_info.channel_count()

    # Check for time correction
    exinfo.stream_time_correction = EEGinlet.time_correction()

    return exinfo


def check_monitor(mywin, exinfo):
    """Checks details about the monitor, and stores in an ExpInfo() object.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
    exinfo : ExpInfo() object
        Object of information about the experiment.

    Returns
    -------
    exinfo : ExpInfo() object
        Object of information about the experiment.
    """

    # Update that monitor check has been run
    exinfo.check_monitor = True

    # Check frame rate and time per frame and save to exinfo
    exinfo.mon_frame_rate = mywin.getActualFrameRate()
    exinfo.mon_ms_frame = mywin.getMsPerFrame()

    return exinfo


def experiment_log(exinfo, run, stim):
    """Writes out a log (txt file) with relevant run information.

    Parameters
    ----------
    exinfo : ExpInfo() object
        Object of information about the experiment.
    run : RunInfo() object
        Object of information / parameters to run task trials.
    stim : Stim() object
        Object of all stimuli for the task.
    """

    ## Set up & open file for experiment log
    logfilename = "PATH_TO_SAVE_FILE_TO" + str(exinfo.subnum) + '_ExpLog.txt'
    logfile = open(logfilename, 'w')

    ## Write to file
    # Basic Information
    logfile.write('\n RUN INFORMATION \n')
    logfile.write('Run Version: ' + str(exinfo.runversion) + '\n')
    logfile.write('Date of Current Version: ' + str(exinfo.dateversion) + '\n')
    logfile.write('Subject Number: ' + str(exinfo.subnum) + '\n')
    logfile.write('Date: ' + str(exinfo.datetimenow) + '\n')

    # Software Information
    logfile.write('\n SOFTWARE INFORMATION \n')
    logfile.write('Pylsl Protocol Version: ' + str(exinfo.lsl_protocol_version) + '\n')
    logfile.write('Pylsl Library Version: ' + str(exinfo.lsl_library_version) + '\n')

    # EEG Information
    logfile.write('\n EEG INFORMATION \n')
    logfile.write('EEG System: ' + str(exinfo.eegsystem) + '\n')
    logfile.write('Re-referenced: ' + str(run.reref) + '\n')
    logfile.write('Channels Used Online: ' + str(run.chans) + '\n')
    logfile.write('Set Sampling Rate: ' + str(run.srate) + '\n')
    logfile.write('Number Samples Used Online: ' + str(run.nsamples) + '\n')

    # Monitor / display information
    if exinfo.check_monitor:
        logfile.write('\n MONITOR INFORMATION \n')
        logfile.write('Monitor Frame Rate: ' + str(exinfo.mon_frame_rate) + '\n')
        logfile.write('Monitor ms per frame: ' + str(exinfo.mon_ms_frame) + '\n')

    # Stream Information
    if exinfo.check_stream:
        logfile.write('\n STREAM INFORMATION \n')
        logfile.write('Stream Name: ' + str(exinfo.stream_name) + '\n')
        logfile.write('Stream Type: ' + str(exinfo.stream_type) + '\n')
        logfile.write('Stream Sampling Rate: ' + str(exinfo.stream_srate) + '\n')
        logfile.write('Stream Number of Channels: ' + str(exinfo.stream_nchans) + '\n')
        logfile.write('Time Correction: ' + str(exinfo.stream_time_correction) + '\n')

    # Experiment Information - THESE FIELDS MAY NEED UPDATING
    logfile.write('\n EXPERIMENT INFORMATION \n')
    logfile.write('Number of exp blocks: ' + str(exinfo.nblocks) + '\n')
    logfile.write('Number trials per block: ' + str(exinfo.ntrials_per_block) + '\n')
    logfile.write('Number of label blocks: ' + str(exinfo.nlabel_blocks) + '\n')
    logfile.write('Number of label trials per block: ' + str(exinfo.ntrials_label_block) + '\n')
    logfile.write('Thresholding - Number Reversals: ' + str(exinfo.nreversals) + '\n')
    logfile.write('Thresholding - Step Sizes: ' + str(exinfo.step_sizes) + '\n')
    logfile.write('Thresholding - Min Number Trials: ' + str(exinfo.nthresh_trials) + '\n')

    # Presentation Information - ADD FIELDS FOR EXPERIMENT SPECIFIC PRESENTATION INFORMATION
    logfile.write('\n PRESENTATION INFORMATION \n')

    # Close log file
    logfile.close()

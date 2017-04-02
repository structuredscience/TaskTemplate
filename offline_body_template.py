from psychopy import visual, core, data, event
import random
import datetime
import numpy as np
import pylsl

"""
THIS IS A TEMPLATE FOR AN OFFLINE EXPERIMET.
PARTS IN ALL CAPS ARE NOTES ON THE TEMPLATE, AND NEED UPDATING TO RUN.

Classes & Functions to run the .... experiment.

Notes:
    - Here, set up to use LSL for sending event markers. This can be changed.
"""

#####################################################################################
################################ Experiment- Classes ################################
#####################################################################################

class ExpInfo(object):
    """Class to store experiment run parameters."""

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
        self.step_sizes = [] # LIST OF STEP SIZES - INTS OR FLOATS

        # Settings for train
        self.ntrain_1 = int()      # Number of trials in first set of practice trials
        self.ntrain_2 = int()      # Number of trials in second set of practice trials

        # Check current lsl version
        self.lsl_protocol_version = pylsl.protocol_version()
        self.lsl_library_version = pylsl.library_version()

        # Set booleans for what have been checked
        self.check_monitor = False


    def update_block_number(self):
        """Increment block number after running a block."""
        self.block_number += 1


class RunInfo(object):
    """Class to store information details used to run individual trials."""

    def __init__(self):

        # Trial stimuli settings - ADD EXPERIMENT SPECIFIC VARIABLES HERE
        self.wait_time = float()        # Time to wait for a response (in seconds)
        self.iti = float()              # Time to wait between trials (in seconds)

        # Initialize file name vars
        self.dat_fn = None
        self.dat_f = None

        # Initialize clock var
        self.clock = None


    def make_files(self, subnum):
        """Initialize data files and write headers."""

        # Experiment Data File
        self.dat_fn = "PATH_TO_SAVE_FILE_TO" + str(subnum) + "_exp.csv"
        self.dat_f = open(self.dat_fn, 'w')
        self.dat_f.write('PUT HEADERS HERE')
        self.dat_f.close()


    def make_clock(self):
        """Make the clock to use for the experiment."""
        self.clock = core.Clock()


class Stim(object):
    """Class to store all the stimuli used in the experiment."""

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
        """Update stimuli. Used in behavioural thresholding.

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


############################################################################
############################## rtPB Functions ##############################
############################################################################


def run_block(mywin, marker_outlet, exinfo, run, stim):
    """Runs a blocks of trials in the experiment.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
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
    """

    # Get block settings
    ntrials_block = exinfo.ntrials_per_block
    block_num = exinfo.block_number

    # Set up matrix for block data
    exp_block_data = np.zeros(shape=(ntrials_block, LEN_TRIAL_DAT_OUT))

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
    for trial in xrange(0, ntrials_block):
        trial_info = [block_num, ETC] # <- LIST OF INFORMATION TO RUN THE TRIAL
                        #  FOR EXAMPLE: [BLOCK_NUM, TRIAL_NUM, RUN_TYPE, SIDE, ETC.]
        trial_dat = run_trial(mywin, marker_outlet, run, stim, trial_info)
        exp_block_data[trial, :] = trial_dat
        iti = run.iti + random.random()  # Adds jitter to ITI
        core.wait(iti)

    # Send marker for end of block
    marker_outlet.push_sample(pylsl.vectorstr(["End Block"]))

    # End of block message
    disp_text(mywin, message, 'That is the end of this block of trials.')

    # Return block data
    return exp_block_data


def make_block_trials(ntrials_block):
    """Creates a matrix of pseudo-random balanced trial parameters for a block of trials.

    Parameters
    ----------
    ntrials_block : int
        Number of trials in the block.

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


def run_trial(mywin, marker_outlet, run, stim, trial_info):
    """This function runs experiment trials.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
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
    """

    # Get index object
    i = Inds()

    # Get trial parameters
    # PULL OUT TRIAL SETTINGS FROM TRIAL_INFO
    #  FOR EXAMPLE, TRIAL_TYPE, SIDE etc.

    # Get fixation cross
    fix = stim.fix

    # Set up trial, pull out stimuli to use for the trial
    ## USE INPUT TRIAL PARAMETERS TO PULL OUT REQUIRED STIM

    # Run the trial
    fix.draw()
    # DRAW ANY TRIAL MARKERS
    mywin.flip()
    marker_outlet.push_sample(pylsl.vectorstr(["Markers"]))
    core.wait(WAIT_TIME + random.random()/10.)

    # Present stimuli to subject
    # DRAW STIMULI
    # FLIP SCREEN
    pres_time = run.clock.getTime()    # In seconds

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

    # Output data - trial data
    trial_dat = [] # <- SET LIST OF DATA YOU WANT TO COLLECT / RETURN

    ## Print data to files
    strlist_exp = [] # <- MAKE A LIST OF STRINGS OF SAME DATA FOR FILE
    strlist_exp = ",".join(strlist_exp)
    dat_f = open(run.dat_fn, 'a')
    dat_f.write(strlist_exp + '\n')
    dat_f.close()

    return trial_dat


def train(mywin, marker_outlet, exinfo, run, stim):
    """Trains the participant on the task.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
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
    train_meth_data : 2d array
        Matrix of method data from train trials.
    """

    # Get index object
    i = Inds()

    # Get number of practice trials per practice block
    nPrac_1 = exinfo.ntrain_1
    nPrac_2 = exinfo.ntrain_2

    # Set up trial parameters
    block_num = -1
    trial_num = 0
    trial_info = [block_num, trial_num, ETC] # <- TRIAL RUN SETTINGS FOR PRACTICE TRIALS

    # Set up and display text to explain the task
    message = visual.TextStim(mywin, text='')
    disp_text(mywin, message, "INSTRUCTIONS")

    # Run a Test Trial to show an example
    disp_text(mywin, message, "Lets try a test trial.")
    initial_trial = run_trial(mywin, marker_outlet, run, stim, trial_info)
    core.wait(1.0)

    # Initialize matrices for train data (train block 1)
    train_exp_1 = np.zeros(shape=(nPrac_1, LEN_TRIAL_DAT_OUT))

    # Run a practice block trials
    disp_text(mywin, message, "Lets try a few more practice trials.")

    for trial in xrange(0, nPrac_1):

        # Run trial
        train_exp_1[trial, :] = run_trial(mywin, marker_outlet, run, stim, trial_info)
        core.wait(1.0)

    # Present more instructions IF NEEDED
    disp_text(mywin, message, "MORE INSTRUCTIONS")

    # Initialize matrices for train data (train block 2)
    train_exp_2 = np.zeros(shape=(nPrac_2, LEN_TRIAL_DAT_OUT))

    # Run another practice block of trials
    for trial in xrange(0, nPrac_2):

        # Run trial
        train_exp_2[trial, :] = run_trial(mywin, marker_outlet, run, stim, trial_info)
        core.wait(1.0)

    # Collect all train data together
    train_exp_data = np.vstack([initial_trial, train_exp_1, train_exp_2])

    # Pause to ask the subject if they have any questions about the task
    str_message = ("If you have any questions about the task "
                   "please ask the experimenter now.")
    disp_text(mywin, message, str_message)

    return train_exp_data


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


def threshold_staircase(mywin, marker_outlet, exinfo, run, stim):
    """Run a psychophysical staircase to set stimuli parameters.
    Here - looking to set parameters for 50 percent detection rate.

    Parameters
    ----------
    mywin : psychopy Window object
        Pointer to psychopy visual window.
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
    stim : Stim() object
        Stim object updated with thresholded lums.
    """

    # Get index object
    i = Inds()

    # Set up trial information for thresholding trials
    block_num = 0
    trial_num = 1
    trial_info = [block_num, trial_num, ETC] # <- TRIAL RUN SETTINGS FOR THRESH TRIALS

    # Set up staircase condition parameters
    stair_conds = [] # <- LIST OF DICTIONARIES TO INITIALIZE STAIRCASES

    # Set up staircase handler
    staircases = data.MultiStairHandler(
        stairType='simple', conditions=stair_conds,
        nTrials=exinfo.nthresh_trials)

    # Initialize matrices to store thresholding data
    thresh_exp_data = np.zeros([0, LEN_TRIAL_DAT_OUT])

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
        stim.update_stim(THRESH_PARAM)

        # Run trial
        exp_trial_dat = run_trial(mywin, marker_outlet, run, stim, trial_info)

        # Inter-trial interval
        core.wait(run.iti + random.random())

        # Add subject response to staircase
        staircases.addResponse(exp_trial_dat[RESP_INDEX])

        # Add data to thresh data matrices
        thresh_exp_data = np.vstack([thresh_exp_data, exp_trial_dat])

        # Increment trial number
        trial_num += 1

        # Take a break after a certain number of trials
        if trial_num % 48 == 0:
            disp_text(mywin, message, break_message)

    # Pull out results from staircases
    #  DEPENDING HOW YOU ARE USING STAIRCASES, THIS MIGHT REQUIRED AVERAGING OVER
    #    MULTIPLE STAIRCASES, AND/OR A SET OF REVERSAL POINTS IN EACH STAIRCASE
    UPDATE_PARAMS = None

    # Update stim to be used for main experiment
    stim.update_stim(UPDATE_PARAMS)

    # Save staircase object
    staircases.saveAsPickle("PATH_TO_SAVE_FILE_TO" + str(exinfo.subnum) + "_thresh")

    return thresh_exp_data, thresh_meth_data, stim


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

    # Monitor / display information
    if exinfo.check_monitor:
        logfile.write('\n MONITOR INFORMATION \n')
        logfile.write('Monitor Frame Rate: ' + str(exinfo.mon_frame_rate) + '\n')
        logfile.write('Monitor ms per frame: ' + str(exinfo.mon_ms_frame) + '\n')

    # Experiment Information - THESE FIELDS MAY NEED UPDATING
    logfile.write('\n EXPERIMENT INFORMATION \n')
    logfile.write('Number of exp blocks: ' + str(exinfo.nblocks) + '\n')
    logfile.write('Number trials per block: ' + str(exinfo.ntrials_per_block) + '\n')
    logfile.write('Thresholding - Number Reversals: ' + str(exinfo.nreversals) + '\n')
    logfile.write('Thresholding - Step Sizes: ' + str(exinfo.step_sizes) + '\n')
    logfile.write('Thresholding - Min Number Trials: ' + str(exinfo.nthresh_trials) + '\n')

    # Presentation Information - ADD FIELDS FOR EXPERIMENT SPECIFIC PRESENTATION INFORMATION
    logfile.write('\n PRESENTATION INFORMATION \n')

    # Close log file
    logfile.close()

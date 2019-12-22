# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:43:55 2015

@author: mhibert
"""
import re   
import os   
"""
re
    used to split each line of the par file by the location of tabs
    into a list with each item corresponding to a column of the par
    file, each item in the list is a string
os
    used to delete the output file if any incorrect button presses
    are found
"""

def check_codes(f1, f2, good_codes, bad_codes):
    '''This function will read through an input par file, replace the numbers 
    for the raw button presses with the condition number in a new output file, 
    and report if and where any incorrect buttons were pressed.
    
    Parameters:
    -----------
    f1: Input par file with raw button presses generated during the task.
    f2: Destinantion and name of the desired output file with the replaced
        button numbers.
    good_codes: A dictionary with the raw button press number (30-33) and the 
                new condition number to replace it with.  
                For SyCAbs: {'2':'1','3':'2','4':'2','30' : '3', '33' : '4'}
                For Stroop: {'31' : '1', '32' : '2', '33' : '3'}
    bad codes: A list with the raw button press numbers for the buttons which 
                should not have been pressed during the task.
                For SyCAbs: ['31', '32']
                FOr Stroop: ['30']'''
                
    inpt = open(f1, 'r')
    outpt = open(f2, 'w')  # creates the new output file
    inpt_lst = inpt.readlines()  # reads the input file into a list
    numlines_inpt = len(inpt_lst)  # length of original file
    neg_event = 0
    for ln in inpt_lst:    
        lst = re.split(r'\t+', ln.rstrip('\n'))  # splits each line by the 
        # location of tabs and removes the new line at the end
        for i in range(len(lst)):  # converts items in the good_codes 
            # dictionary to the new value and writes all values to the output
            # par file separated by tabs
            if lst[i] in good_codes:
                print(lst)
                print(str(lst[i]) + ' was replaced with...')
                lst[i] = good_codes[lst[i]]
                print(lst[i])
                outpt.write(lst[i] + '\t')
            else:
                outpt.write(lst[i] + '\t')
        if not lst[0].startswith('#'):
            if float(lst[2]) < 0:
                print("Negative event duration detected at time " + lst[0])
                neg_event += 1                
        outpt.write('\n')
    inpt.close()
    outpt.flush()
    outpt.close()

    outpt = open(f2, 'r')  # reopen the output par file as read only

    outpt_lst = outpt.readlines()
    numlines_outpt = len(outpt_lst)  # length of the generated output file

    print('There are %s lines in the input file and %s lines \
        in the output file.' % (str(numlines_inpt), str(numlines_outpt)))
    #check to make sure both file are the same length

    wrong_buttons = 0

    for i in range(len(outpt_lst)):
        lst = re.split(r'\t+', outpt_lst[i])  # split each line of the output 
        # file as before
        for a in range(len(lst)):
            if lst[a] in bad_codes:  # check for wrong button presses and 
                # report where they are and the total
                print('Incorrect button pressed in line %s' % (str(i + 1)))
                wrong_buttons += 1

    print('There were %s incorrect button presses.' % (str(wrong_buttons)))
    print('There were '+str(neg_event)+' negative event durations detected.')
    outpt.close()

    if wrong_buttons >= 1:  # if there were any wrong buttons pressed which 
        # should not have been pressed the output file is deleted
        os.remove(f2)

def mem(f1, f2, expected_trials = 50, perf_check = True, settings = 
                         {'bad_codes' : {'29':'3', '32':'3', '33':'3'}, 
                          'buttons' : {'30':'3', '31':'3'}, 
                          'correct_resp':{'1':('30', '1'), '2':('31', '2')},
                          'incorrect_resp':{'1':('31', '4'), '2':('30', '4')},
                          'no_resp':{'0':'4', '1':'4', '2':'4'}}):
    '''
    This function will check the accuracy of a subject's responses for the 
    memory task and re-categorize the triggers as follows:
    1: correct recognition of old word
    2: correct recognition of new word
    3: all button presses
    4: all incorrect responses including false positives and negatives, double
       button presses, and trials without a response    
    
    Parameters:
    f1: str
        The full path to the location of the original par file with the 
        subject's responses to input into the script.
    f2: str
        The full path to the location of the desired output par file.
    expected_trials: int
        The number of trials there should be in the par file. This number
        defaults to 50.
    perf_check: bool
        Whether the script should perform a performance check on the subject's
        responses at the end, and rename the output par file if the subject 
        answered fewer than 70% of trials correctly.  This value defaults to 
        True.
    settings: dictionary
        The settings for the script giving the original codes in the input par
        file and the values they will be changed to.  The dictionary defaults
        to:
        {'bad_codes' : {'29':'3', '32':'3', '33':'3'}, 
         'buttons' : {'30':'3', '31':'3'}, 
         'correct_resp':{'1':('30', '1'), '2':('31', '2')},
         'incorrect_resp':{'1':('31', '4'), '2':('30', '4')},
         'no_resp':{'0':'4', '1':'4', '2':'4'}}
    '''
    
    # inpt = open(f1, 'r')
    # outpt = open(f2, 'w')
    # inpt_lst = inpt.readlines()
    # lns = []
    # run_info = inpt_lst[0]  # header of original par file
    # for ln in inpt_lst:
    #     if not ln.startswith('#'):  # remove commented lines in orig par file
    #         lst = re.split(r'\t+', ln.rstrip('\n'))  # split by tabs
    #                                                  # remove new line at end
    #         lns.append(lst)
    # for l in range(2):  # hack to deal with end of the events list
    #     lns.append(['9999.9999', '999', '0000.0000', '0.0', 'fake'])
    # trials = 0
    # correct = 0
    # incorrect = 0
    # correct_hit = 0
    # correct_reject = 0
    # incorrect_hit = 0
    # incorrect_reject = 0
    # no_response = 0
    # wrong_button = 0
    # double_button = 0
    # unexpected_resp = 0
    # for i in range(len(lns)):
    #     if lns[i][1] == '999':  # don't read the hack lines at the end
    #         break
    #     elif lns[i][1] in settings['bad_codes']:
    #         # check for incorrect button press and change the code
    #         print 'Wrong button pressed at time %s' % lns[i][0]
    #         wrong_button += 1
    #         lns[i][1] = settings['bad_codes'][lns[i][1]]
    #     elif lns[i][1] in settings['correct_resp']:  # check for trial start
    #         trials += 1
    #         if lns[i+1][1] in settings['buttons'] and \
    #         lns[i+2][1] in settings['buttons']:  # two responses to one trial
    #             # even if first is a correct response, the trial will be
    #             # categorized as an incorrect response
    #             double_button += 1
    #             print 'Double button press at time %s' % lns[i+1][0]
    #             lns[i][1] = settings['incorrect_resp'][lns[i][1]][1]
    #         elif settings['correct_resp'][lns[i][1]][0] == lns[i+1][1]:
    #             # only one response and correct response in next line
    #             correct += 1
    #             if lns[i][1] == '1':
    #                 correct_hit += 1
    #             elif lns[i][1] == '2':
    #                 correct_reject += 1
    #             lns[i][1] = settings['correct_resp'][lns[i][1]][1]
    #         elif settings['incorrect_resp'][lns[i][1]][0] == lns[i+1][1]:
    #             # only one response and incorrect response in next line
    #             incorrect += 1
    #             if lns[i][1] == '1':
    #                 incorrect_hit += 1
    #             elif lns[i][1] == '2':
    #                 incorrect_reject += 1
    #             lns[i][1] = settings['incorrect_resp'][lns[i][1]][1]
    #         elif lns[i+1][1] in settings['no_resp']:
    #             # no response to stimulus before next trial begins
    #             # the next trial can either be another word or fixation
    #             print 'No response at time %s' % lns[i][0]
    #             no_response += 1
    #             lns[i][1] = settings['no_resp'][lns[i][1]]
    #     elif lns[i][1] in settings['buttons']:
    #         # change the code for all correct button presses
    #         lns[i][1] = settings['buttons'][lns[i][1]]
    #         if lns[i-1][1] == '0':
    #             # check if button is pressed during fixation
    #             print 'Response given when not expected at time %s' % lns[i][0]
    #             unexpected_resp += 1
    # rprt =('#total correct = ' + str(correct) + '\n' + '#total incorrect = ' +
    #        str(incorrect) + '\n' + '#correct hit = ' + str(correct_hit) +'\n'+
    #        '#correct reject = '+ str(correct_reject) +'\n'+'#false positive = '
    #        + str(incorrect_hit) + '\n' + '#false negative = ' +
    #        str(incorrect_reject) +'\n'+ '#no response = ' + str(no_response) +
    #        '\n' + '#wrong button = ' + str(wrong_button) + '\n' +
    #        '#multiple button presses = ' + str(double_button) + '\n' +
    #        '#unexpected responses = ' + str(unexpected_resp))
    # pct_correct = float(correct)/float(trials) * 100
    # print 'Subject answered %s of trials correctly.' % pct_correct
    # print rprt
    # inpt.close()
    # outpt.write(run_info)
    # if not trials == expected_trials:  # check for correct number of trials
    #     warn=('#Warning, an incorrect number of trials were found. %s trials'+
    #           ' were expected but there were %s trials.') % (expected_trials,
    #                                                          trials)
    #     outpt.write(warn + '\n')
    #     print warn
    # outpt.write(rprt + '\n')
    # for x in range(2):  # remove hack lines added above
    #     lns.remove(lns[-1])
    # for ln in lns:
    #     for i in ln:
    #         outpt.write(i + '\t')
    #     outpt.write('\n')
    # outpt.close()
    # if perf_check is True and float(correct)/float(trials) < .7:
    #     # check subject performance and rename the output par file if
    #     # performance was poor
    #     split = os.path.split(f2)
    #     os.rename(f2, os.path.join(split[0], 'bad_responses_' + split[1]))
    #     print ("Output file" + f2 + " renamed because subject answered fewer "
    #            "than 70% of trials correctly. Replace the default settings "
    #            "parameter with the following and retry: "
    #            "{'bad_codes' : {'29':'3', '32':'3', '33':'3'}, "
    #             "'buttons' : {'30':'3', '31':'3'}, "
    #             "'correct_resp':{'1':('30', '1'), '2':('31', '2')}, "
    #             "'incorrect_resp':{'1':('31', '1'), '2':('30', '2')}, "
    #             "'no_resp':{'0':'0', '1':'1', '2':'2'}}")
    pass


def sycabs(f1, f2,
           settings = {'bad_codes' : {'29':'5', '31':'5', '32':'5', '34':'5'}, 
                       'buttons' : {'30':'3', '33':'4'}, 
                       'correct_resp':{'1':('30', '1'), '2':('33', '1'),
                                       '3':('33', '2'), '4':('30', '2')},
                       'incorrect_resp':{'1':('33', '1'), '2':('30', '1'),
                                         '3':('30', '2'), '4':('33', '2')},
                       'no_resp':{'0':'0','1':'1','2':'1','3':'2','4':'2',
                                  '999':'999'}}):
    '''
    This function will check the accuracy of a subject's responses for the 
    sycabs task and re-categorize the triggers as follows:
    1-4: remain the same, i.e. words and symbols
    5: left button press
    6: right button press
    7: a press of an incorrect button i.e. the "thumb" button or the middle 2
       buttons, NOT an incorrect response    
    
    Parameters:
    f1: str
        The full path to the location of the original par file with the 
        subject's responses to input into the script.
    f2: str
        The full path to the location of the desired output par file.
    expected_trials: int
        The number of trials there should be in the par file. This number
        defaults to 72.
    settings: dictionary
        The settings for the script giving the original codes in the input par
        file and the values they will be changed to.  The dictionary defaults
        to:
           settings = {'bad_codes' : {'29':'5', '31':'5', '32':'5', '34':'5'}, 
                       'buttons' : {'30':'3', '33':'4'}, 
                       'correct_resp':{'1':('30', '1'), '2':('33', '1'),
                                       '3':('33', '2'), '4':('30', '2')},
                       'incorrect_resp':{'1':('33', '1'), '2':('30', '1'),
                                         '3':('30', '2'), '4':('33', '2')},
                       'no_resp':{'0':'0','1':'1','2':'1','3':'2','4':'2',
                                  '999':'999'}}):

    '''

    warn = ''
    print('{} -> {}'.format(f1, f2))
    inpt = open(f1, 'r')
    outpt = open(f2, 'w')
    inpt_lst = inpt.readlines()
    lns = []
    run_info = inpt_lst[0]  # header of original par file
    for ln in inpt_lst:
        if not ln.startswith('#'):  # remove commented lines in orig par file
            lst = re.split(r'\t+', ln.rstrip('\n'))  # split by tabs
                                                     # remove new line at end
            lns.append(lst)
    for l in range(2):  # hack to deal with end of the events list
        lns.append(['9999.9999', '999', '0000.0000', '0.0', 'fake'])
    trials = 0
    correct = 0
    incorrect = 0
    no_response = 0
    wrong_button = 0
    double_button = 0
    unexpected_resp = 0
    #lns2 = [] # hack for when one button on button box was broken
    for i in range(len(lns)):
        if lns[i][1] == '999':  # don't read the hack lines at the end
            break
        elif lns[i][1] in settings['bad_codes']:  
            # check for incorrect button press and change the code
            print('Wrong button pressed at time %s' % lns[i][0])
            wrong_button += 1
            lns[i][1] = settings['bad_codes'][lns[i][1]]
        elif lns[i][1] in settings['correct_resp']:  # check for trial start
            trials += 1
            if lns[i+1][1] in settings['buttons'] and \
            lns[i+2][1] in settings['buttons']:  # two responses to one trial
                # even if first is a correct response, the trial will be
                # categorized as an incorrect response
                double_button += 1
                print('Double button press at time %s' % lns[i+1][0])
                # lns[i][1] = settings['incorrect_resp'][lns[i][1]][1]
            elif settings['correct_resp'][lns[i][1]][0] == lns[i+1][1]:
                # only one response and correct response in next line
                correct += 1
                lns[i][1] = settings['correct_resp'][lns[i][1]][1]
            elif settings['incorrect_resp'][lns[i][1]][0] == lns[i+1][1]:
                # only one response and incorrect response in next line
                incorrect += 1
                lns[i][1] = settings['incorrect_resp'][lns[i][1]][1]
            elif lns[i+1][1] in settings['no_resp']:
                # no response to stimulus before next trial begins
                # the next trial can be a word, symbols, or fixation
                print('No response at time %s' % lns[i][0])
                no_response += 1
                lns[i][1] = settings['no_resp'][lns[i][1]]
                #lns2.append(i) # hack for when one button on button box was broken
        elif lns[i][1] in settings['buttons']:
            # change the code for all correct button presses
            lns[i][1] = settings['buttons'][lns[i][1]]
            if lns[i-1][1] == '0':  
                # check if button is pressed during fixation
                print('Response given when not expected at time %s' % lns[i][0])
                unexpected_resp += 1
    for ln in lns:
        if float(ln[2]) < 0:
            warn += "!!! Warning!!! Negative event duration at time "+ln[0]+'!!!'
            print(warn)
    # if not trials == expected_trials:  # check for correct number of trials
    #     warn=('#Warning, an incorrect number of trials were found. %s trials'+
    #           ' were expected but there were %s trials.') % (expected_trials,
    #                                                          trials)
    #     outpt.write(warn + '\n')
    #     print(warn)
    # hack for when one button on button box was broken
    #lns3 = [] 
    #for i in range(len(lns2)): 
    #    lns3.append(lns2[i] + i) 
    #for i in lns3:    
    #    lns.insert(i+1, [str(float(lns[i][0])+1), '3', '0.1000', '1.0', 'fake_1!'])
    rprt =('#total correct = ' + str(correct) + '\n' + '#total incorrect = ' + 
           str(incorrect) + '\n' + '#no response = ' + str(no_response) +
           '\n' + '#wrong button = ' + str(wrong_button) + '\n' + 
           '#multiple button presses = ' + str(double_button) + '\n' + 
           '#unexpected responses = ' + str(unexpected_resp))
    pct_correct = float(correct)/float(trials) * 100.
    print('Subject answered %s of trials correctly.' % pct_correct)
    print(rprt)
    inpt.close()
    outpt.write(run_info)
    outpt.write('# setings = ' + str(settings) + '\n')
    outpt.write(rprt + '\n')
    for x in range(2):  # remove hack lines added above
        lns.remove(lns[-1])
    for ln in lns:
        for i in ln:
            outpt.write(i + '\t')
        outpt.write('\n')
    outpt.close()
    return warn
    
    

    
    
    
    
    
    
    
    
    
    
    
    

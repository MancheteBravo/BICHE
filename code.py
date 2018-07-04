"""
Created on Sun Jun 17 15:25:28 2018
PLAY ULTIMATE CHICKEN HORSE
MUST BE IN CHALLENGE
MUST PLAY THE SHEEP CARACTER
MUST BE IN WINDOWED 1920 x 1080 resolution
MUST BE PATIENT as it runs on you CPU right now

THIS IS A DQN Algorithm using only the screen's input and playing one of 9 possible moves allowed for BICHE.
This is a frankenstein algorithm based on the DQN tutorial on the pytorch website that plays cartpole using only the screen inputs
(https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html),
the keras AI that plays flappy birds by Ben Lau (https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html), 
SethBling's mari/o and mariflow alroithms (https://www.youtube.com/channel/UC8aG3LDTDwNR1UQhSn9uVrw)
and SerpentAI's magnificent framework(https://github.com/SerpentAI, http://serpent.ai/).

@author: EDOUARD DESJARDINS
"""

import os
import time
import win32api, win32con
from PIL import ImageOps, Image,ImageGrab
import numpy as np
import skimage
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T
import random
from collections import namedtuple
from itertools import count
import math
import matplotlib
import matplotlib.pyplot as plt

"""
I believe that you could put a dtype = cuda call here to change all tensors of your GPU
"""
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")    


################ TO CONTROLE THE MOUSE BUTTONS ################


def leftClick():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
    print ('LEFT CLICK')

def leftDOwn():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(.1)
    print ('HOLD LEFT CLICK')

def leftUp():
     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
     time.sleep(.1)
     print ('RELEASE LEFT CLICK')
     

def rightClick():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
    time.sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
    print ('RIGHT CLICK')

def rightDOwn():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
    time.sleep(.1)
    print ('HOLD RIGHT CLICK')

def rightUp():
     win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
     time.sleep(.1)
     print ('RELEASE RIGHT CLICK')

########## MOUSE MOUVEMENTS #######################

##### use this to put the mouse somewhere on the screen before doing a action ####
def mousePos(cord):
    win32api.SetCursorPos((cord[0], cord[1]))
    
###### use this to get the numbered coordinates of the pixels where the mouse is at the moment this is used #####

def get_cords():
    x,y = win32api.GetCursorPos()
    print (x,y)
    
################ KEYBOARD- ALL BUTTONS AND A FUNCTION TO CALL THEM EASILY #####################
    #Giant dictonary to hold key name and VK value
    ## Here I used ChrisKiehl's wonderfull library (https://gist.github.com/chriskiehl/2906125)
VK_CODE = {'backspace':0x08,
           'tab':0x09,
           'clear':0x0C,
           'enter':0x0D,
           'shift':0x10,
           'ctrl':0x11,
           'alt':0x12,
           'pause':0x13,
           'caps_lock':0x14,
           'esc':0x1B,
           'spacebar':0x20,
           'page_up':0x21,
           'page_down':0x22,
           'end':0x23,
           'home':0x24,
           'left_arrow':0x25,
           'up_arrow':0x26,
           'right_arrow':0x27,
           'down_arrow':0x28,
           'select':0x29,
           'print':0x2A,
           'execute':0x2B,
           'print_screen':0x2C,
           'ins':0x2D,
           'del':0x2E,
           'help':0x2F,
           '0':0x30,
           '1':0x31,
           '2':0x32,
           '3':0x33,
           '4':0x34,
           '5':0x35,
           '6':0x36,
           '7':0x37,
           '8':0x38,
           '9':0x39,
           'a':0x41,
           'b':0x42,
           'c':0x43,
           'd':0x44,
           'e':0x45,
           'f':0x46,
           'g':0x47,
           'h':0x48,
           'i':0x49,
           'j':0x4A,
           'k':0x4B,
           'l':0x4C,
           'm':0x4D,
           'n':0x4E,
           'o':0x4F,
           'p':0x50,
           'q':0x51,
           'r':0x52,
           's':0x53,
           't':0x54,
           'u':0x55,
           'v':0x56,
           'w':0x57,
           'x':0x58,
           'y':0x59,
           'z':0x5A,
           'numpad_0':0x60,
           'numpad_1':0x61,
           'numpad_2':0x62,
           'numpad_3':0x63,
           'numpad_4':0x64,
           'numpad_5':0x65,
           'numpad_6':0x66,
           'numpad_7':0x67,
           'numpad_8':0x68,
           'numpad_9':0x69,
           'multiply_key':0x6A,
           'add_key':0x6B,
           'separator_key':0x6C,
           'subtract_key':0x6D,
           'decimal_key':0x6E,
           'divide_key':0x6F,
           'F1':0x70,
           'F2':0x71,
           'F3':0x72,
           'F4':0x73,
           'F5':0x74,
           'F6':0x75,
           'F7':0x76,
           'F8':0x77,
           'F9':0x78,
           'F10':0x79,
           'F11':0x7A,
           'F12':0x7B,
           'F13':0x7C,
           'F14':0x7D,
           'F15':0x7E,
           'F16':0x7F,
           'F17':0x80,
           'F18':0x81,
           'F19':0x82,
           'F20':0x83,
           'F21':0x84,
           'F22':0x85,
           'F23':0x86,
           'F24':0x87,
           'num_lock':0x90,
           'scroll_lock':0x91,
           'left_shift':0xA0,
           'right_shift ':0xA1,
           'left_control':0xA2,
           'right_control':0xA3,
           'left_menu':0xA4,
           'right_menu':0xA5,
           'browser_back':0xA6,
           'browser_forward':0xA7,
           'browser_refresh':0xA8,
           'browser_stop':0xA9,
           'browser_search':0xAA,
           'browser_favorites':0xAB,
           'browser_start_and_home':0xAC,
           'volume_mute':0xAD,
           'volume_Down':0xAE,
           'volume_up':0xAF,
           'next_track':0xB0,
           'previous_track':0xB1,
           'stop_media':0xB2,
           'play/pause_media':0xB3,
           'start_mail':0xB4,
           'select_media':0xB5,
           'start_application_1':0xB6,
           'start_application_2':0xB7,
           'attn_key':0xF6,
           'crsel_key':0xF7,
           'exsel_key':0xF8,
           'play_key':0xFA,
           'zoom_key':0xFB,
           'clear_key':0xFE,
           '+':0xBB,
           ',':0xBC,
           '-':0xBD,
           '.':0xBE,
           '/':0xBF,
           '`':0xC0,
           ';':0xBA,
           '[':0xDB,
           '\\':0xDC,
           ']':0xDD,
           "'":0xDE,
           '`':0xC0}

def press(*args):
    '''
    one press, one release.
    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.05)
        win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)

def pressAndHold(*args):
    '''
    press and hold. Do NOT release.
    accepts as many arguments as you want.
    e.g. pressAndHold('left_arrow', 'a','b').
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.05)
           
def pressHoldRelease(*args):
    '''
    press and hold passed in strings. Once held, release
    accepts as many arguments as you want.
    e.g. pressAndHold('left_arrow', 'a','b').

    this is useful for issuing shortcut command or shift commands.
    e.g. pressHoldRelease('ctrl', 'alt', 'del'), pressHoldRelease('shift','a')
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.05)
            
    for i in args:
            win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)
            time.sleep(.1)
            
        

def release(*args):
    '''
    release depressed keys
    accepts as many arguments as you want.
    e.g. release('left_arrow', 'a','b').
    '''
    for i in args:
           win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)

"""
EXAMPLE of a function that would automatically strat the game and go
in the stage for you. It is not necessary anymore but it is fun to have
"""
def startGame():
    #Premier click
    mousePos((816,656))
    leftClick()
    time.sleep(2)
    
    #deuxième click
    mousePos((456,292))
    leftClick()
    leftClick()
    time.sleep(4)#attend que ca load
    
    #troisieme click (sélectionne le chicken
    mousePos((1400,570))
    leftClick()
    time.sleep(2)
    
    pressAndHold('d')
    print ('VA A DROITE')
    
    press('spacebar')
    print ('SAUTE')
    time.sleep(3)
    
    press('spacebar')
    print ('SAUTE')
    time.sleep(2)
    
    release('d')
    print('STOP ALLER A DROITE')
    
    pressAndHold('a')
    print ('VA A GAUCHE')
    time.sleep(1)
    press('spacebar')
    print ('SAUTE')
    
    time.sleep(0.7)
    release('a')
    print('STOP ALLER A GAUCHE')
    
"""
This section defines every possible move used in the game. Yes they are in french
but you just need to know that gauche means left and droite means right. Free french lesson
in bonus! The tiempo variable is to determine how long she presses the buttons. You can ajust this.
"""
tiempo = 0.3
def bougeGauche():
    pressAndHold('a')
    time.sleep(tiempo)
    release('a')
    print('ALLER A GAUCHE')
    
def bougeDroite():
    pressAndHold('d')
    time.sleep(tiempo*2)
    release('d')
    print('ALLER A DROITE')
    
def runGauche():
    pressAndHold('a','shift')
    time.sleep(tiempo)
    release('a','shift')
    print('RUN A GAUCHE')
    
def runDroite():
    pressAndHold('d','shift')
    time.sleep(tiempo*2)
    release('d','shift')
    print('RUN A DROITE')
    
def jump():
    pressAndHold('spacebar')
    time.sleep(tiempo)
    release('spacebar')
    print('SAUTE')
    
def jumpDroit():
    pressAndHold('spacebar', 'd')
    time.sleep(tiempo*2)
    release('spacebar','d')
    print('SAUTE A DROITE')
    
def jumpGauche():
    pressAndHold('spacebar', 'a')
    time.sleep(tiempo)
    release('spacebar','a')
    print('SAUTE A GAUCHE')
    
def crouch():
    pressAndHold('s')
    time.sleep(tiempo)
    release('s')
    print('CROUCH')
    
def dance():
    pressAndHold('r')
    time.sleep(tiempo)
    release('r')
    print('DANCE')
    
def wallJumpDroite():
    pressAndHold('d')
    press('spacebar')
    time.sleep(tiempo)
    release('d')
    print('WALL JUMP on RIGHT WALL')
    
def wallJumpGauche():
    pressAndHold('w')
    press('spacebar')
    time.sleep(tiempo)
    release('w')
    print('WALL JUMP on LEFT WALL')
    
    
#################### THIS IS THE LIST OF ACTIONS SHE WILL  BE ABLE TO DO ###########
    
    
actions = [bougeGauche, bougeDroite, runGauche, runDroite, jump, jumpDroit, jumpGauche, dance]


"""
OK so this image extracted as image_ref is actually a reference image of what the screen looks like 
when you win. She needs it to compare every frame she sees with it (actually, just the section where the sheep appears 
on the end game screen) and see if she has finished the level and won or lost.
"""



image_ref = torch.load('C:\\Users\\User\\Dropbox\\Python Scripts\\reference__1529885704_this_win.pt') 

### choose the section of the image she will look at to see if she is finished ####

image_ref_fin = torch.split(image_ref,10,dim=0)
image_ref_fin = image_ref_fin[2]
image_ref_fin = torch.split(image_ref_fin,5,dim=1)
image_ref_fin = image_ref_fin[4]

### choose the section of the image she will look at to see if she has won ####

image_ref_win = torch.split(image_ref,5,dim=0)
image_ref_win = image_ref_win[4]
image_ref_win = torch.split(image_ref_win,4,dim=1)
image_ref_win = image_ref_win[3]


######################## NEURAL NETWORK A.K.A. HER BRAIN #################################
"""
this selects the transitons which is a move and associates it a initial state(etat_init), the move action she did (action), 
the next state she's in (etat_final) and the reward associated with the move (reward).
"""       
Transition = namedtuple('Transition', 
                        ('etat_init','move','etat_final','reward')) 


"""
What follows is a cyclic buffer that allows her to remember transitions that she did previously. the .sample() function
help select a RANDOM batch of previous move to learn with since this helps strenghten the DQN algorithm

"""

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity #maximum capacity of her memory
        self.memory=[] #initialise an empty ensemble to contain this memory
        self.position = 0  #initialise at position 0, so first memory
        
    def push(self, *args):
        """ Saves a transition """
        if len(self.memory) < self.capacity: #If there is room in the memory
            self.memory.append(None) #add an empty memory
        self.memory[self.position] = Transition(*args) #fill it with the corresponding transition namedtuple seen at line 442
        self.position = (self.position + 1) % self.capacity # move up one memory spot and see if there is still room 
        
    def sample(self, batch_size): ###To take a random batch of memories according to the batch_size variable determined later
        return random.sample(self.memory, batch_size)
    
    def __len__(self):    #### to see the size of the memory
        return len(self.memory)
 

       

"""
THIS is the neural network. It has 3 convoluted layers taking 1 input which is the image and one dense (linear)
layer that finishes with the 9 possible ouptupts with a weigh value added to it. She will use the value with the higest score.
Batchnorm normalises the output of every convoluted layer to minimise calculations. We then use ReLU activation function 
for they are easier to compute, specially if you have a shitty PC like me. This activation function 
computes the weight of every transitions used (connections between nodes). some back propagation of this will occur further down
Note that the matrixes (yeah yeah i know i should use the word tensors but screw it)
need to have matching dimensions. Because linear algebra man.
"""     

class Brain(nn.Module):
    
    def __init__(self):

        super(Brain,self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.output = nn.Linear(448, 8)
        
    def forward(self,x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) 
        return self.output(x.view(x.size(0), -1))
    

   
"""
The eyes of BICHE where the image becomes  x,y in tensor[y,x]
it becomes black and white, the resolution is completey dimed down and normalized.
You end up with a tensor  of 1x40x80 with values between 0 and 1, 0 being black and 1 being white.
This is how she sees

"""

def grab(show = False, save = False, png = False):

    im = ImageOps.grayscale(ImageGrab.grab())
    im = ImageOps.fit(im,(80,40))
    if show == True:
        im.show() ########## just in case you want to slow down everything, you can input show=True
    a=skimage.img_as_float(im)
    a=np.float32(a)
    pytorch_frame = torch.FloatTensor(torch.from_numpy(a))
    ##### To save your own reference image as seen on line 420
    if save == True:
        torch.save(pytorch_frame,os.getcwd() + '\\reference__' + str(int(time.time())) +'.pt') 
    #### To save it as a png
    if png == True:
        im.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) +'.png', 'PNG') 

    return pytorch_frame.unsqueeze(0).to(device)## formating

BATCH_SIZE = 128   ### amount of memories she will look back to learn (YOU CAN PLAY WITH THIS VALUE)
"""
constants used in the exponential decay for exploration vs exploitation. The more she makes moves, the more likely 
she will used a learned move vs a completely random move. YOU CAN PLAY WITH THIS VALUE
"""

gamma = 0.999  
eps_start = 0.9  
eps_end = 0.05
eps_decay = 15000
target_update = 10

"""
2 instances of the brain. Policy is the one that works and target is the one that learns. Note that i used the same 
variable names as the pytorch tutorial

"""
policy_net = Brain().to(device)
target_net = Brain().to(device)
target_net.load_state_dict(policy_net.state_dict()) 
##### this copies the parameters and buffers extracted with the state_dict function from policy instance to target instance
##### it is the next generation of the brain


target_net.eval() ## She evaluates these parameters

#### she optimises and looks in the memory, this potimiser will be seen later in the code
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


"""
HERE is where she chooses is she does a new move or an old move acording to a decaying exponential. 
If the random number is higher, she does a learned move, if not, she does a random move.
"""
number_of_moves = 0 ## move counter

def select_move(etat_init):
    global number_of_moves
    sample = random.random() ##random nuber between 0 et 1
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1.* number_of_moves/eps_decay) ## exponentielle
    number_of_moves += 1
    if sample > eps_threshold: 
        with torch.no_grad(): ### if you are wondering what this does, it just calculates the move it would do (learned) but it doesn,t calculate the gradient, which will not back propagate in the neural network
            print('OLD MOVE')
            return policy_net(etat_init).max(1)[1]

 
    else:
        print('NEW MOVE')
        return torch.tensor(([[random.randrange(7)]]), device = device, dtype=torch.long) #### make a random move



"""
Here we train her. 
She looks at the data provided by each step (move she makes), she puts the data in the right format, she calculates the Q state she's in
and the next one and she calculates the error between the expect Q states value and the real one observed. To have a clearer view of this algorithm, 
google DQN, you should be fine

"""



def optimize_model():
    if len(memory) < BATCH_SIZE: #### if not enough moves in the memory yet, calm down and keep playing
        return
    transitions = memory.sample(BATCH_SIZE)
    ########### this transposes the transition saved so has to match the size of the matrixestensorzzzz
    batch = Transition(*zip(*transitions))
    ###### COmpute the mask and concatenate the tensor(puts it back to back instead of stacked)
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.etat_final)), device=device, dtype=torch.uint8)
    non_final_mask = torch.cat((non_final_mask,non_final_mask,non_final_mask),0)
    non_final_etat_final = torch.cat([s for s in batch.etat_final if s is not None])
    etat_init_batch = torch.cat(batch.etat_init) ### split the tesnors transition according to their category

    reward_batch = torch.cat(batch.reward)


    #### calculate Q(s_t,a) - the model's Q(s_t) and then choose the move column

    etat_init_action_value = policy_net(etat_init_batch)

    #### calculate V(s_(t+1)) for subsequent states
    etat_final_values = torch.zeros(BATCH_SIZE*3, device = device) ## so this part is just to stack three inputs together for matching the sizes
    etat_final_values[non_final_mask] = target_net(non_final_etat_final).max(1)[0].detach() ### detach clears the memory
    etat_final_values = etat_final_values[0:BATCH_SIZE]
    ##### with V, find Q expected at the next step
    expected_etat_init_action_value = (etat_final_values*gamma) + reward_batch

    etat_init_action_value=etat_init_action_value[0:BATCH_SIZE].max(1)
    #### calculate the error using Huber error which is like squared mean error but with less weight given to outliers

    
    etat_init_action_value = etat_init_action_value[1].float() 
    expected_etat_init_action_value = expected_etat_init_action_value.float()
    loss = F.smooth_l1_loss(etat_init_action_value,expected_etat_init_action_value)    
    loss = torch.autograd.variable(loss, requires_grad =True) ### besoin d'UN GRADIENT

    """
    this next part optimizes the model by backl propagation of the gradients. It changes the weight of
    every neural network connections implicated in the move selection to make it more likely if the reward is positive or
    less likely if the reward is negative.
    """
    optimizer.zero_grad()
    loss.backward() 
    torch.nn.utils.clip_grad_norm(policy_net.parameters(),1)
    optimizer.step()
    

nombre_episode = 2000 #### number of GAMES PLAYED
data = []
    
for i_episode in range(nombre_episode):
    print('episode =  ' + str(i_episode))
    finished_episode = False ### this variable says if she is playing or not. it changes if it sees the end game screen as extracted at line 420
    ## click on the screen because you are in windowed mode and every time you will start the algorithm it has to press back with the mose on the screen
    mousePos((816,656))
    leftClick()
    time.sleep(0.1)
    
     ###### look at the game BICHE! with your beautiful brown eyes

    etat_init = grab()
    etat_init = etat_init.reshape(1,1,etat_init.shape[1],etat_init.shape[2])
    etat_init = torch.cat((etat_init,etat_init,etat_init),0)

    bonus_right = 0
    for t in count(): #number of moves done in the episode
        ########## fait une action
        action = select_move(etat_init)
        action = action[0]
        move_number=action.item()
        move = actions[move_number]() ## this is to take just the position of the move in the action list instead of the name of the move (line 409)
"""
this part is to explain to her that she need to go right. so choose a level where she has to go right only
These values need to be tweeked. If the time bias is too strong, she will kill herself willingly quickly so it is less penalising the trying the level.
If going right is too rewarding, she will stick her stupid head on a wall and move right a million times before finishing the level
Soooo yeah....
"""
        if move_number == (5):
            bonus_right += 0.1
        if move_number in (1,3):
            bonus_right += 0.05
        if move_number in (0,2,6):
            bonus_right -= 0.05
            
        time_bias = t/500 ### THIS IS THE TIME FACTOR WHICH PENALISES HER FOR TAKING TOO LONG, this needs to be tweeked
        reward = 0. + bonus_right ###### Every move you had the bonus for goign right and malus for going left
        print(reward)
        reward = torch.tensor([reward], device=device)

        ######## observe lthe new state

        next_screen = grab()
        next_screen = next_screen.reshape(1,1,next_screen.shape[1],next_screen.shape[2])## to match tensor sizes (this algorithm uses 3x1x40x80 inputs)
        next_screen = torch.cat((next_screen,next_screen,next_screen),0)
        test_screen = grab()
        #### Look if theepisode is finished

        verify_if_stopped = torch.split(test_screen,10,dim=1)
        verify_if_stopped = verify_if_stopped[2]
        verify_if_stopped = torch.split(verify_if_stopped,5,dim=2)
        verify_if_stopped = verify_if_stopped[4]
        


        if torch.ByteTensor.all(torch.eq(verify_if_stopped,image_ref_fin)) == True : ## if it matches the end screen
           

            time.sleep(6)
            test_screen = grab()
            verify_if_won = torch.split(test_screen,5,dim=1)
            verify_if_won = verify_if_won[4]
            verify_if_won = torch.split(verify_if_won,4,dim=2)
            verify_if_won = verify_if_won[3]

            ############### Take ref image and compare
            
            finished_episode = True
            
            if torch.ByteTensor.all(torch.eq(verify_if_won,image_ref_win)) == True :
                reward = 4. - time_bias + bonus_right
                reward = torch.tensor([reward], device=device)
                print('WIN')
            else:
                reward = -1. - time_bias + bonus_right
                reward = torch.tensor([reward], device=device)
                print('LOSE')

            time.sleep(4)
            mousePos((434,246))### these positions are to click on the RETRY button top left, you might have to modify these values using get_cords() line 85)
            leftClick() 
            time.sleep(.5)
            mousePos((434,246))
            leftClick()
            time.sleep(.1)
               
          
        if finished_episode == False:
            etat_final = next_screen
                

        elif finished_episode == True:
            etat_final = None

        else : ## this is just in case something is incredibly wrong, the code should never go here
            print('Probleme chum')
            etat_final = None
                
        if finished_episode == True:
            data.append((i_episode,reward.item(),time_bias,t+1))
            print(data[i_episode]) 
            ### ok so data saves all the info on the run. But it is just a variable. WHEN YOU FINISH' YOU NEED TO MANUALLY
            ### SAVE THIS OR ELSE YOU WILL HAVE DONE EVERYTHING FOR NOTHING. I should save it automatically and write a .txt file
            ### as it progresses. soon

            break
    """
    so if the episode is not finished, she keeps going.
    She puts the moves/reward in her memory, she makes the new screen she saw the old screen, she optimizes (back scattering the gradients)
    and she does it all over again
    """

            ##### storage in the memory
        memory.push(etat_init,action,etat_final,reward)
            ##### next move
        etat_init = etat_final

            #### optimise
        optimize_model()


    
            
            #### updtae the neural network after a couple of time (like an autosave of the nn)
    if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print('YOU LEARNDED')
            
            
print('FINISHED')        


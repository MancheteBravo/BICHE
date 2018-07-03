"""
Created on Sun Jun 17 15:25:28 2018
PLAY ULTIMATE CHICKEN HORSE
MUST BE IN CHALLENGE

@author: EDOUARD DESJARDINS
"""
#import pyscreenshot as ImageGrab
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



################ pour contrôler la souris ################

## juste un click normal GAUCHE
def leftClick():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
    print ('LEFT CLICK TABARNAK')
    ## tient la souris enfoncé
def leftDOwn():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(.1)
    print ('TIENS LE PITON GAUCHE DE LA SOURIS')
    ## lache le piton de gauche
def leftUp():
     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
     time.sleep(.1)
     print ('LÄCHE LE PITON GAUCHE DE LA SOURIS')
     
## juste un click normal DROIT
def rightClick():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
    time.sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
    print ('RIGHT CLICK TABARNAK')
    ## tient la souris enfoncé
def rightDOwn():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
    time.sleep(.1)
    print ('TIENS LE PITON DROIT DE LA SOURIS')
    ## lache le piton de gauche
def rightUp():
     win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
     time.sleep(.1)
     print ('LÄCHE LE PITON DROIT DE LA SOURIS')

########## MOUVEMENT DE SOURIS ####
def mousePos(cord):
    win32api.SetCursorPos((cord[0], cord[1]))
     
def get_cords():
    x,y = win32api.GetCursorPos()
    print (x,y)
    
################ CLAVIER #####################
    #Giant dictonary to hold key name and VK value
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


############### POUR PARTIR LE JEUX ##############
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
    
##################### ACTION LIST ######################################
tiempo = 0.3
def bougeGauche():
#    mousePos((816,656))
#    leftClick()r
#    time.sleep(0.1)
    pressAndHold('a')
    time.sleep(tiempo)
    release('a')
    print('ALLER A GAUCHE')
    
def bougeDroite():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1) 
    pressAndHold('d')
    time.sleep(tiempo*2)
    release('d')
    print('ALLER A DROITE')
    
def runGauche():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('a','shift')
    time.sleep(tiempo)
    release('a','shift')
    print('RUN A GAUCHE')
    
def runDroite():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('d','shift')
    time.sleep(tiempo*2)
    release('d','shift')
    print('RUN A DROITE')
    
def jump():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('spacebar')
    time.sleep(tiempo)
    release('spacebar')
    print('SAUTE')
    
def jumpDroit():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('spacebar', 'd')
    time.sleep(tiempo*2)
    release('spacebar','d')
    print('SAUTE A DROITE')
    
def jumpGauche():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('spacebar', 'a')
    time.sleep(tiempo)
    release('spacebar','a')
    print('SAUTE A GAUCHE')
    
def crouch():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('s')
    time.sleep(tiempo)
    release('s')
    print('CROUCH')
    
def dance():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('r')
    time.sleep(tiempo)
    release('r')
    print('DANCE')
    
def wallJumpDroite():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1)
    pressAndHold('d')
    press('spacebar')
    time.sleep(tiempo)
    release('d')
    print('WALL JUMP on RIGHT WALL')
    
def wallJumpGauche():
#    mousePos((816,656))
#    leftClick()
#    time.sleep(0.1) 
    pressAndHold('w')
    press('spacebar')
    time.sleep(tiempo)
    release('w')
    print('WALL JUMP on LEFT WALL')
    
    
#################### fait un move random ###########
    
    
actions = [bougeGauche, bougeDroite, runGauche, runDroite, jump, jumpDroit, jumpGauche, dance]


########### pour avoir al référence de quand une partie est finies #############

#image_ref_test = torch.load('C:\\Users\\Edouard\\Dropbox\\Python Scripts\\reference__1529537989-win.pt')  #### BUREAU

image_ref = torch.load('C:\\Users\\User\\Dropbox\\Python Scripts\\reference__1529885704_this_win.pt') ## MAISON

image_ref_fin = torch.split(image_ref,10,dim=0)
image_ref_fin = image_ref_fin[2]
image_ref_fin = torch.split(image_ref_fin,5,dim=1)
image_ref_fin = image_ref_fin[4]

image_ref_win = torch.split(image_ref,5,dim=0)
image_ref_win = image_ref_win[4]
image_ref_win = torch.split(image_ref_win,4,dim=1)
image_ref_win = image_ref_win[3]

"""
PLAY BUT NO BRAIN
"""
#
#def play():
#    random_move_position = random.randint(0,8)
#    move = actions[random_move_position]()
#    
#    
#    return move
    ### pendant qu'il peut continuer a jouer, il joue
         ### choisi une action au hasard
         ### si meurt, fait le click click et recommence à jouer
##
##    
#number_of_moves = 0
#mousePos((816,656))
#leftClick()
#time.sleep(0.1)
#
#while number_of_moves < 2000:
#    
#    play()
#    x1 = grab()
#    x2 = torch.split(x1,10,dim=0)
#    x2=x2[8]
#    x2 = torch.split(x2,50,dim=1)
#    x2=x2[1]
##    time.sleep(9)
#
#    
#    if torch.ByteTensor.all(torch.eq(x2,image_ref_test)) == True :
#           
#        print ('YOU DIED OR YOU WON')
#        time.sleep(8)
#        mousePos((434,246))
#        leftClick() 
#        time.sleep(.1)
#        mousePos((434,246))
#        leftClick()
#        time.sleep(.1)
#        continue
#    number_of_moves+= 1
#    
#    
    
  # set up matplotlib
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
#    from IPython import display
#
#plt.ion()
#  

    
    
    
######################## NEURONES #################################

       
"""
sur quoi tu le fait rouler, cpu ou gpu
"""
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")        
        
        
Transition = namedtuple('Transition', 
                        ('etat_init','move','etat_final','reward')) #représente une action et son nanane à la fin

"""
ce qui suit est un buffer cyclique permettant de retenir les transitions récentes. Il a aussi une fonction .sample() pour 
sélectionner une batch random de transitions pour le training ce qui stabilise l'algorithme DQN
"""

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity #le max qui peut rentrer dans la mémoire??
        self.memory=[] #initialise un ensemble vide pour contenir la mémoire
        self.position = 0  #iniitialise la mémoire au point 0
        
    def push(self, *args):
        """ Sauvegarde une transition """
        if len(self.memory) < self.capacity: #si il reste de la place dans sa mémoire
            self.memory.append(None) #rajout une mémoire vide
#        print('save')
        self.memory[self.position] = Transition(*args) #sauvegarde la transition dans la mémoire au point position
        self.position = (self.position + 1) % self.capacity # augmente la mposition de la mémorie tant qu,il reste d ela place ??
        
    def sample(self, batch_size): ###pour prendre une batch random de mémoires
        return random.sample(self.memory, batch_size)##prend des trucs random dans memory selon la taille de batch size
    
    def __len__(self):    #### pour savoir al taille dece qu'il y aa en mémoire.
        return len(self.memory)
 

       

"""
Ce qui suit est le cerveau ou le réseau de neuronnes
on veut quelques 
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
        """
        ici on fait passer les nodes dans une fonction (sigmoid ou dans ce cas ci relu) 
        afin d'obtenire les poids pour chacune des transitions
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) 
        return self.output(x.view(x.size(0), -1))
    

   
"""
Les yeux de l'algorithme qui transpose l'image  x,y en tensor[y,x]
tu le mets en noir et blanc, tu réduit la résolution, tu le sauvegarde ne matrice puit en tenseur

SERAIT-CE mieux de mettre le tenseur en vecteur??  on verra
"""

def grab(show = False):

    im = ImageOps.grayscale(ImageGrab.grab())
    im = ImageOps.fit(im,(80,40))
    if show == True:
        im.show() ########## pour la voir
    a=skimage.img_as_float(im)
    a=np.float32(a)
    pytorch_frame = torch.FloatTensor(torch.from_numpy(a))
    ##### pour sauvegardé le tenseur pour des références
#    torch.save(pytorch_frame,os.getcwd() + '\\reference__' + str(int(time.time())) +'.pt') 
    #### pour sauvegarder en png
#    im.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) +'.png', 'PNG') 

    return pytorch_frame.unsqueeze(0).to(device)## question de format

BATCH_SIZE = 128   ### nombre de mémoire a processer en même temps
gamma = 0.999  ## dans l'équation
eps_start = 0.9  ### LA PLAGE DE VARIATION DE EPSILON QUI VAUT ENTRE 0 ET 1
eps_end = 0.05
eps_decay = 15000
target_update = 10

"""
fait aller le cerveau policy_net et target_net sont le cerveau (nn.Module) en 2 modes différents

"""
policy_net = Brain().to(device)
target_net = Brain().to(device)
target_net.load_state_dict(policy_net.state_dict()) ##### copie les parametres et buffer de state_dict dans ce module et ses déscendant

target_net.eval() ## se met en mode evaluation pour certains parametres

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


"""
c'est ici qu'il va choisir si il fait un move selon ce qu'il a appris ou un move random (EPSILON) 
c'est une exponentielle décroissante au taux de eps-decay explore vs exploit
"""
number_of_moves = 0 ## compteur de nombre de moves

def select_move(etat_init):
    global number_of_moves
    sample = random.random() ##un chiffre aléatoire entre 0 et 1
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1.* number_of_moves/eps_decay) ## exponentielle
    number_of_moves += 1
    if sample > eps_threshold: #### test pour voir si on fait un nouveau move ou si on garde un vieux
        with torch.no_grad(): ### pas de gradient alors pas de réciprocité arrière dans le réseau SAUVED E LA MÉMOIRE JE MEN CRIS STU
            print('OLD MOVE')
            return policy_net(etat_init).max(1)[1]
#            return torch.tensor(([[3]]), device = device, dtype=torch.long) #### pas trop sur ?????????? 
 
    else:
        print('NEW MOVE')
        return torch.tensor(([[random.randrange(7)]]), device = device, dtype=torch.long) #### fait un move random



"""
MAINTENANT on entraine le AI

ici il regarde les donné de chaques pas, il les met en dans le bon format, il fait le calcul des Q pour le pas 
où il est et le pas suivant et il calcul l'erreur entre le pas suivant attendu et le pas suivant réel
"""



def optimize_model():
    if len(memory) < BATCH_SIZE: #### des fonction dans la mémorie
        return
    transitions = memory.sample(BATCH_SIZE)
#    print(transitions)
    ########### transpose la batch avec zip et les mets ensemble chaque catégories! (verified)
    batch = Transition(*zip(*transitions))
#    print(batch)

    ###### compute un masque d'état non finaux et plug les tenseurs en batch bout à bout (concatenate)
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.etat_final)), device=device, dtype=torch.uint8)
    non_final_mask = torch.cat((non_final_mask,non_final_mask,non_final_mask),0)
#    print(non_final_mask) ####### check si y'a des datas alors met un 1
    non_final_etat_final = torch.cat([s for s in batch.etat_final if s is not None]) ### met tout bout a bout
#    print(non_final_etat_final.size())
#    print(non_final_etat_final) #### met les états finaux bout a bout
    ####### met les états en lignes pris de Transition ligne 629
    
    etat_init_batch = torch.cat(batch.etat_init) ### les splits selon la catédorie

    reward_batch = torch.cat(batch.reward)


    #### calcule Q(s_t,a) - le model calule Q(s_t) et après on choisi la colonne des moves

    etat_init_action_value = policy_net(etat_init_batch)

    #### calcule V(s_(t+1)) pour tous les états suivant
    etat_final_values = torch.zeros(BATCH_SIZE*3, device = device)
    etat_final_values[non_final_mask] = target_net(non_final_etat_final).max(1)[0].detach() ### detach clear la mémoire
    etat_final_values = etat_final_values[0:BATCH_SIZE]
    ##### avec les V, trouve les valeur de Q attendu au pprochain pas
    expected_etat_init_action_value = (etat_final_values*gamma) + reward_batch

    etat_init_action_value=etat_init_action_value[0:BATCH_SIZE].max(1)
    #### CAlcule l'erreur (loss function mais selon Huber loss un peu comme squared mean error mais linéaire a grande erreur

    
    etat_init_action_value = etat_init_action_value[1].float()
    
    expected_etat_init_action_value = expected_etat_init_action_value.float()
    
#    loss = F.smooth_l1_loss(etat_init_action_value,expected_etat_init_action_value.unsqueeze(1))#unsqueeze pour els dimensions
    loss = F.smooth_l1_loss(etat_init_action_value,expected_etat_init_action_value)
    
    loss = torch.autograd.variable(loss, requires_grad =True) ### besoin d'UN GRADIENT


    ### optimize le model (fait varier les poid par gradientdescent(les deltas selon la fonction functionnelle))
    optimizer.zero_grad()
    loss.backward() ##########  no grad param here nut whyyyyyyyyy
    
    torch.nn.utils.clip_grad_norm(policy_net.parameters(),1)
    optimizer.step()
    

nombre_episode = 2000 #### nombre d'épisode
data = []
    
for i_episode in range(nombre_episode):
    print('episode =  ' + str(i_episode))
   ## il est en train de jouer
    finished_episode = False
    ## click sur le screen
    mousePos((816,656))
    leftClick()
    time.sleep(0.1)
    
     ###### check le jeux (ca donne des tenseurs)

    etat_init = grab()
    etat_init = etat_init.reshape(1,1,etat_init.shape[1],etat_init.shape[2])
    etat_init = torch.cat((etat_init,etat_init,etat_init),0)

    bonus_right = 0
    for t in count(): #compte jusqu'a ce que la partie finisse je crois
        ########## fait une action
        action = select_move(etat_init)
        action = action[0]
        move_number=action.item()
        move = actions[move_number]() ###pour prendre juste le chiffre et pas la structure tensorielle

        if move_number == (5):
            bonus_right += 0.1
        if move_number in (1,3):
            bonus_right += 0.05
        if move_number in (0,2,6):
            bonus_right -= 0.05
            
        time_bias = t/500### pour factor in le temps que ça prend de faire le level
        reward = 0. + bonus_right ###### A CHAQUE MOVE AjOUTE UN ZERO DANS REWARD plus son bonus séquentiel!!
        print(reward)
        reward = torch.tensor([reward], device=device)

        ######## observe le nouvel état

        next_screen = grab()
        next_screen = next_screen.reshape(1,1,next_screen.shape[1],next_screen.shape[2])
        next_screen = torch.cat((next_screen,next_screen,next_screen),0)
        test_screen = grab()
        #### check si c'est le screen de fin

        verify_if_stopped = torch.split(test_screen,10,dim=1)
        verify_if_stopped = verify_if_stopped[2]
        verify_if_stopped = torch.split(verify_if_stopped,5,dim=2)
        verify_if_stopped = verify_if_stopped[4]
        
        ############condition sur la fin du jeux

        if torch.ByteTensor.all(torch.eq(verify_if_stopped,image_ref_fin)) == True :
           

                
            time.sleep(6)
            test_screen = grab()
            verify_if_won = torch.split(test_screen,5,dim=1)
            verify_if_won = verify_if_won[4]
            verify_if_won = torch.split(verify_if_won,4,dim=2)
            verify_if_won = verify_if_won[3]

            ############### prend image référence de gagner et compare
            
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
            mousePos((434,246))
            leftClick() 
            time.sleep(.5)
            mousePos((434,246))
            leftClick()
            time.sleep(.1)
               
          
        if finished_episode == False:
            etat_final = next_screen
                

        elif finished_episode == True:
            etat_final = None

        else :
            print('Probleme chum')
            etat_final = None
                
        if finished_episode == True:
            data.append((i_episode,reward.item(),time_bias,t+1))
            print(data[i_episode])

            break
    

            ##### store dans la mémoire
        memory.push(etat_init,action,etat_final,reward)
            ##### bouge au prochain etat dans sa tête
        etat_init = etat_final

            #### fait un pas d'optimisation
        optimize_model()

            #### si la partie est fini, sort la duration de l'épisode
    
            
            #### updtae le reseau de neuronne target
    if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print('YOU LEARNDED')
            
            
print('FENI')        


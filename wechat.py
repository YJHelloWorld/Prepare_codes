import itchat
import threading
import os
k=0



@itchat.msg_register(itchat.content.TEXT)
def chat_trigger(msg):
    
    if msg['Text'] == 'hi':
        txt = os.popen('tail /home/jianyao/pytorch-CycleGAN-and-pix2pix/prepare_codes/nohup.out')
        info = txt.readlines()
	line_1 = info[-1].strip('\r\n')
	line_2 = info[-2].strip('\r\n')
	itchat.send('details are:%s %s'%(line_1, line_2))

itchat.auto_login(enableCmdQR=2,hotReload=True)
threading.Thread(target = itchat.run).start()


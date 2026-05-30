"""Check MoCo experiment status on UIC server."""
import os, paramiko
HOST = os.environ.get("UIC_HOST", "10.250.93.98")
PORT = int(os.environ.get("UIC_PORT", "6422"))
USER = os.environ.get("UIC_USER", "uic2")
PASSWORD = os.environ["UIC_PASSWORD"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

def ssh(cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
    return stdout.read().decode(), stderr.read().decode()

out, err = ssh('grep "ALL DONE\|Exp8 done" /home/uic2/Raymond/moco_nohup.log')
print(out.strip())

out, err = ssh('tail -3 /home/uic2/Raymond/moco_nohup.log')
print('Last:', out.strip()[:300])

out, err = ssh('ls -d /home/uic2/Raymond/results/exp*moco* 2>/dev/null')
print('Dirs:', out.strip())

client.close()

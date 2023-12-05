#necessary imports
import configparser


# configparser object
config = configparser.ConfigParser()
config.read("config.ini")

#user credential
user=config.get("credentials","user")
password=config.get("credentials","password")


def check_credentials(userid: str, clientsecretkey:str):
    if userid is not None and clientsecretkey is not None:
            hdruserid = userid
            hdrclientsecretkey = clientsecretkey
    else:
        return False
    
    if hdruserid==user and hdrclientsecretkey==password:
        return True
    else: 
        return False
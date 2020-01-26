from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'AC468e8e00e4f9472b4851cd77e533cd91'
auth_token = '8b9348526a36abd1852dc554ebbd9e05'
client = Client(account_sid, auth_token)

def send_sms():
    message = client.messages \
        .create(
             body='Suspicious activity has been reported!',
            from_='+13437002917',
            to='+16476086035'
        )
    print(message.sid)

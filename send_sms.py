from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'place_key_here'
auth_token = 'place_key_here'
client = Client(account_sid, auth_token)

def send_sms():
    message = client.messages \
        .create(
             body='Suspicious activity has been reported!',
            from_='+13437002917',
            to='+16476086035'
        )
    print(message.sid)
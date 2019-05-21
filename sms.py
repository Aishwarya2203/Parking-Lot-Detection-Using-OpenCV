from twilio.rest import Client


account_sid = 'AC0d83ee9321a5b7eaea13f951ce2ca503'
auth_token = '3e3a74fb4ab23a21c25cecdc518eb796'
client = Client(account_sid, auth_token)


message = client.messages \
                .create(
                     body="Parking Spot Detected",
                     from_='+12019925335',
                     to='+14243818357'
                 )

print(message.sid)

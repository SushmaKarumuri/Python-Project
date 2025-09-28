import qrcode

input_URL = input("Enter the URL to generate QR code: ")

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=15,
    border=4,
)

qr.add_data(input_URL)
qr.make(fit=True)
import matplotlib.pyplot as plt

# Convert QR code to an image
img = qr.make_image(fill_color="black", back_color="white")

# Show the image in the notebook
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

img = qr.make_image(fill_color="red", back_color="white")
img.save("url_qrcode.png")

print(qr.data_list)
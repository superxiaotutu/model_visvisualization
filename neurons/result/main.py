# with open(  'a' + '.html', 'a') as f:
#     f.write('''  <main%s></main%s>
#                   <script>
#                     var app = new GroupWidget_1cb0e0d({
#                       target: document.querySelector( 'main%s' ),''' % (2, 3, 4))
import base64
a='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'
imgdata = base64.b64decode(a)
with open(  'a.jpg', 'wb') as f_jpg:
    f_jpg.write(imgdata)
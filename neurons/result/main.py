with open(  'a' + '.html', 'a') as f:
    f.write('''  <main%s></main%s>
                  <script>
                    var app = new GroupWidget_1cb0e0d({
                      target: document.querySelector( 'main%s' ),''' % (2, 3, 4))
import io

# In-memory buffer for composing the resulting html file
_buffer = None

class file_writer:

    def begin_html_file(self, title, css_rules=None):
        global _buffer
        _buffer = io.StringIO()
        _buffer.write("""<!DOCTYPE HTML>\n<html>\n<head>\n
                          <title>{}</title>
                          <meta charset="utf-8">
                       """.format(title))
        if css_rules != None:
            _buffer.write('<style>{}</style>'.format(css_rules))
        _buffer.write('</head>\n<body>')
    
    def end_html_file(self, output_fname):
        global _buffer
        _buffer.write('</body>\n</html>')
        with open(output_fname, 'w', encoding='utf-8') as outfile:
            outfile.write(_buffer.getvalue())
        _buffer.close()

    def add_div(self, contents, styleclass=None):
        global _buffer
        c = '' if styleclass == None else ' class="{}" '.format(styleclass)
        _buffer.write('<div {}>{}</div>'.format(c, contents))


class table_writer:
    
    def begin_table(self, styleclass=None, table_headings=()):
        global _buffer
        sclz = '' if styleclass == None else ' class={}'.format(styleclass)
        _buffer.write("<table{}>\n".format(sclz))
        _buffer.write("<thead>\n<tr>")
        for heading in table_headings:
            _buffer.write("<th>{}</th>".format(heading))
        _buffer.write("</tr>\n</thead>\n<tbody>\n")

    def append_table_row(self, row_contents=()):
        global _buffer
        _buffer.write("<tr>")
        for content in row_contents:
           _buffer.write("<td>{}</td>".format(content))
        _buffer.write("</tr>\n")


    def end_table(self):
        global _buffer
        _buffer.write("</tbody>\n</table>")

class image_writer:
    
    def add_image(self, src, alt=None, width=None, height=None, styleclass=None):
        global _buffer
        _buffer.write(self.get_image(src, alt, width, height))

    def get_image(self, src, alt=None, width=None, height=None, styleclass=None):
        global _buffer
        w = '' if width == None else ' width={} '.format(width)
        h = '' if height == None else ' height={} '.format(height)
        a = '' if alt == None else ' alt="{}" '.format(alt)
        c = '' if styleclass == None else ' class="{}" '.format(styleclass)
        return '<img src="{}"{}{}{}{}/>'.format(src, a, c, w, h)

class text_writer:

    def add_heading(self, text, level=1, styleclass=None):
        global _buffer
        c = '' if styleclass == None else ' class="{}" '.format(styleclass)
        if level < 1 or level > 6:
            raise ValueError("Invalid heading level {}".format(level))
        _buffer.write('<h{}{}>{}</h{}>'.format(level, c, text, level))

    def add_paragraph(self, text, styleclass=None):
        global _buffer
        c = '' if styleclass == None else ' class="{}" '.format(styleclass)
        _buffer.write('<p{}>{}</p>'.format(c, text))

    def begin_list(self, ordered=False, styleclass=None):
        global _buffer
        o = 'o' if ordered == True else 'u'
        c = '' if styleclass == None else ' class="{}" '.format(styleclass)
        _buffer.write('<{}l{}>'.format(o, c))

    def add_list_item(self, itemtext, styleclass=None):
        global _buffer
        c = '' if styleclass == None else ' class="{}" '.format(styleclass)
        _buffer.write('<li{}>{}</li>'.format(c, itemtext))

    def end_list(self, ordered=False):
        global _buffer
        o = 'o' if ordered == True else 'u'
        _buffer.write('</{}l>'.format(o))

    def add_link(self, href="#", link_contents=None, styleclass=None):
        global _buffer
        _buffer.write(self.get_link(href, link_contents, styleclass))

    def get_link(self, href="#", link_contents=None, styleclass=None):
        c = '' if styleclass == None else ' class="{}" '.format(styleclass)
        t = '' if link_contents == None else link_contents
        return '<a href="{}"{}>{}</a>'.format(href, c, t)



def decode_html(encoded):
    if encoded is None:
        return ""
    else:
        return encoded.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt", ">").replace("&amp;", "&")


def html_to_plain(html, return_segments=False):
    cursor = 0
    html2 = ""
    while cursor < len(html):
        cha = html[cursor]
        if cha == "<":
            remaining = html[cursor+1:]
            closing_ind = remaining.find(">")
            if closing_ind > 0:
                if not (remaining.startswith("span") or remaining.startswith("/span") or remaining.startswith(
                        "font") or remaining.startswith("/font") or remaining.startswith(
                        "strong") or remaining.startswith("/strong")):
                    html2 += html[cursor:cursor + closing_ind + 2]
                cursor += closing_ind + 2
            else:
                html2 += html[cursor:]
                break
        else:
            opening_ind = html[cursor + 1:].find("<")
            if opening_ind > 0:
                html2 += html[cursor:cursor + opening_ind + 1]
                cursor += opening_ind + 1
            elif opening_ind == 0:
                cursor += 1
            else:
                html2 += html[cursor:]
                break

    cursor = 0
    segments = list()
    nnewline = 0
    nparam = 0
    while cursor < len(html2):
        cha = html2[cursor]
        if cha == "<":
            remaining = html2[cursor+1:]
            closing_ind = remaining.find(">")
            if closing_ind > 0:
                if remaining.startswith("div") or remaining.startswith("tr"):
                    nnewline += 1
                elif remaining.startswith("/div") or remaining.startswith("/tr"):
                    if nnewline > 0:
                        segments.append("\n" * nnewline)
                        nnewline = 0
                elif remaining.startswith("br>") or remaining.startswith("br/>") or remaining.startswith("br />"):
                    segments.append("\n")
                elif remaining.startswith("p") or remaining.startswith("h1") or remaining.startswith(
                        "h2") or remaining.startswith("h3") or remaining.startswith("h4") or remaining.startswith("h5"):
                    segments.append("\n")
                    nparam += 1
                elif remaining.startswith("/p") or remaining.startswith("/h1") or remaining.startswith(
                        "/h2") or remaining.startswith("/h3") or remaining.startswith("/h4") or remaining.startswith("/h5"):
                    segments.append("\n"*nparam)
                    nparam = 0
                cursor += closing_ind + 2
            else:
                break
        else:
            opening_ind = html2[cursor + 1:].find("<")
            if opening_ind > 0:
                segments.append(decode_html(html2[cursor:cursor+opening_ind+1]))
                segments.append("\n" * nparam)
                nparam = 0
                cursor += opening_ind + 1
            elif opening_ind == 0:
                cursor += 1
            else:
                segments.append(decode_html(html2[cursor:]))
                break

    if return_segments:
        return segments
    else:
        return "".join(segments).strip('\n')


from requests import get

API_CALL = "https://api.opencitations.net/index/v2/references/doi:10.1103/PhysRevD.76.013009"
HTTP_HEADERS = {"authorization": "ef7cdb02-4d10-4579-9abc-342c14c2f26c-1760178072"}

print(get(API_CALL, headers=HTTP_HEADERS).content)
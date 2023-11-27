# import necessary libraries
from insightface.app import FaceAnalysis
import cv2
import os 
import configparser
import base64
import numpy as np
import time
# imports from local file
from elastic import index_data ,search_elastic

# reading config file
config = configparser.ConfigParser()
config.read("config.ini")

model_start=time.time()
# defining the insightface model
model = FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(480, 480))

print(f"model time {time.time()- model_start}")

# image preprocessing
def image_pre_process(image_path:str or bool=False , bs64:str or bool=False):
    if bs64==False:
        original_image = cv2.imread(image_path)
        # print(original_image.shape)
        # #original_image= cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        decoded_image = base64.b64decode(bs64)
        np_arr = np.frombuffer(decoded_image, dtype=np.uint8)
        original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    denoised_image = cv2.fastNlMeansDenoisingColored(original_image, None, 10, 10, 7, 15)
    return denoised_image

# image directory example
image_dir=config.get("image","image_dir")

# create bulk embeddings
def create_bulk_embeddings(image_dir:str):
    
    list_images=os.listdir(image_dir)

    inside_start=time.time()
    i=1
    for image in list_images:
        image_path=os.path.join(image_dir,image)
        print(f"image_path=========> {image_path}")
        
        denoised_image=image_pre_process(image_path, bs64=False)
        faces =model.get(denoised_image)

        for face in range(len(faces)):

            data={
                "embeddings":faces[face]["embedding"].tolist(),
                "image_path":image_path   
            }
            
            try:
                print(index_data(data,"face_search"))
                print(f"Number=====>{i}")
                i+=1
                
            except Exception as e:
                print(f"Number=====>{i}")
                i+=1
                print("Exception ======>",e)
                # will log exception 
                return "Certain Error has occured. Check logs"
            
    print(f"inside_final {time.time() -inside_start}")
    return "Data indexed successfully"
                
def single_embedding_creation(img_bs64:str, image_name:str):
    denoised_image= image_pre_process(img_bs64,image_path=False)

    faces =model.get(denoised_image)

    for face in range(len(faces)):

        data={
            "embeddings":faces[face]["embedding"].tolist(),
            "image_name":image_name   
        }
        
        try:
            print(index_data(data,index_name="face_verification"))
            print(f"Number=====>{i}")
            i+=1
            
        except Exception as e:
            print(f"Number=====>{i}")
            i+=1
            print("Exception ======>",e)
            # will log exception 
            return "Certain Error has occured. Check logs"
    
    return "Data indexed successfully"


# search functionality

def search(bs64:str):
    
    # doing pre processing
    denoised_image= image_pre_process(bs64=bs64, image_path=False)
    # getting image embeddings
    target_embeddings =model.get(denoised_image)
    
    # converting to list to search in elastic
    embeddings=target_embeddings[0]["embedding"].tolist()

    # threshold 
    threshold = float(config.get("image","threshold"))

    output=search_elastic(target_embeddings=embeddings, similarity_threshold=threshold)

    return output


if __name__=="__main__":
    overall_start=time.time()
    result=search(bs64="/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBUVFRgVFRUYGRgaGhkYGhgZGBgaGBocGRwaGhgcGRgcIS4lHB4rIRkYJjgmKy8xNTU1GiQ7QDs0Py40NTEBDAwMEA8QGhISGjQhISQ0NDE0NDQ0NDQ0NDE0NDQxNDE0NDQ0NDQ0NDQ0NTQ0NDQxNDQ0NDQ0MT80PzRANDQ0Mf/AABEIAQQAwgMBIgACEQEDEQH/xAAcAAACAgMBAQAAAAAAAAAAAAADBAIFAAEGBwj/xAA+EAABAgMFBgQFAgUEAQUAAAABAAIDBBESITFB8AVRYXGBkQYiobETMsHR4QdCI1JikvEzcoLCFBU0Q1PS/8QAGQEAAwEBAQAAAAAAAAAAAAAAAAECAwQF/8QAIREBAQEBAAIDAQADAQAAAAAAAAECESExAxJBUQQiMhP/2gAMAwEAAhEDEQA/APTihORXITkiFg/KpBalx5VJASaptC01SCaRmoQHmRWoY+ZAMtUloLYKZxJYhPitbiQOdy02ZYcHNPUJKGWKLSDeFJAYq6Z+Y9FYqvmfmPRBVBqMEJqMEJQcnGYBKOTbcAg8pJSPim0rMYpwUnEUApvUEyYsW1iQbchFFehKVjS/yqajLC5Eogm2qYCiERqaRGodfMiWgBUmgXl/jjxwWudBli0AEB7z5nEkgWWMz68BTGgcjrfEXjWWlB532nZNbeT25rzfaX6nTkwXNlmCGy8WzeQLz5jeGmg4rlojmAmNMPcXEVDalz3YFtThUVqAPlBFTgDVze13vpZHw2CllrScsycylarOerKanYz3fx5mI8m8+Z9nfhddmEvLwYR+YV5+gNVXy7CSruRlm1/m11U3cjXPxWrSUhsHyPjMN1Cx5bSlw+V7aZZrpJDa86z/AEtoF9LgyZh2gTh/qG8crRwKoGMFKWaaqE1DiXcVH/vGt/xnUwv1CmoH/u5Vrm3eeE4jhWjqg38RiM8eh2T4slpt38N9HGnkeLLxcMsDuqCV5/CiuFwJocRfQ8xvxQ4uzWvNto81xqLn3VvBGOIV5+TOmOvh1l7Awo7VxXhHbLy74MRxfd5Hn5rhUh2/HFdq0q2NRcmxglXJsIEYlphMpeOgUk9QARHqATJqixbosQG3oRRXIZUrGlcESihK4IhQTAiAIYVb4g2q2XhOeTgCeQGJ1v4Jhy/6keLRLtMFjvMQbZBIOVGgi++6pqKXcV4Y/aLy61U1vpebq4051TXiParpmM6IXE14mhIJvAJ48MzQVVVDbUoVmd8GHOLjacSTx9hwRpeCXX0uWS8Cpv1wVvLwlhvfHXj4w4EsrmQl8LkOXgq4lYC59atdEzIIyCs+Gm2MWFqniuhwmJ6Xbel2NvT8uxXn2z16EmZZwHxYTrEUA0cM+Br07BKbD/UeIyJ8KbAp8toNo4EYk0oCOFB1zuIbK3Ljv1B8PUYJiGMKW6D1XXnTi3mPaZeYa9oc0ggioINQeqsAvH/0r8SlwEu8/wCypwObeWfQ7wvYArYRiBHR0COgUnEQ0WIhhMmli2sQEnIRCM5CKlQstgiIcrgiIDYC8V/Vnb9t3wGk4gGhustvp1JF2F2RAXs0w+y0kYr5a8QTRiR3vqSK0bXGjbq0yqQT1TEVlU3KMqk1bbPZcp3eRt8U7o7AhUT8CGgQWq0lIa47eu7Pg1JwVcQGJWAAE8xwSkForWKJasZejshVVcTajCZmn4CXcKIzXJ5idXp2EQmo0FsRjmOvDgQeqSYE9CNy3zXPp49JsfKTL2XgsfVp/wBpqO4ovo7ZM4I0GHFGD2h3I5jvVeJeO5OxHbEGLmtJ4kEj2+q9Q/TiIXSEInfEA5B7gPt0W345/wBdSgxkZBjJClIiEixEIJpYsW6LEwk5BcjOQioULK4IqFK4IqYCmIVppbvBHcEL5U2tCc2M9rvmDnAjcamo7r6wIXiP6u+GfhxP/KZ8sR3mFMHUcXHrQHqUCPMArnZouVOr2QFGhZ/L6dHwf9LCCy8K3l2gBVAi0vOS1D2oK4Fc0za6rqR00IlPQQaKjkdph19kkbwKjur6QmGHE0O5V9KX3ybl2XXphh3KUZwDQQQpiCbNQnJxNvSjol95opfFa0XuHUpGZ2ZUm06gJv6fKBv/ACgu2O3ChuzcaHtiFckRrV/HQwJptLr1bSxBwXJysqQQ22DQYA5ZZLo9mEimGIbWt3CquRlquf8AH0GvwgBUkOpnWjm0HqF6T4Y2eYErBhO+Zrau/wBziXu9XFcX4mexszJueaMD3vJvr5HMdcBeTWlwXo0GKHtDmmoN4K0ZX+iIUZFQ4qE0m9CCK9CCaWLFixMJOQnIrkNyhQkrgUVClcCipwNtXI/qpJfE2fENB5KPruoaE/2krr2rybx7tuLFjRIIivbBYSywwltst8ri8i91SHCzWlALlOtTM6v48XV5HjbhQq6lnCyLxgFuf2eYXnYSDiRw5LoZfb0RzWF7Yb6ipJZZN4yskAX8FnqzWetsZudWVVMhl5DRhmVay0ixu5v9V1rvl0W27SBiF4gsqWfKb2kiyK+Wya0qb6qUOC97nF7KBwIFioDa/wAuJ6JZkn6vVt/FvLwZYijnFzwMQ9xPqVZysCF+1x5O+xVZsKRdBJdUP3Nc0ForcSa3nlWinHYGuJv5ClOgpcFd5/Wcmv2LDaLX0a5gJbXIVHP0ornZ7wYeI6kBcXMPqMr3YccRSvRdLsphLbNaVGSzvitZ258k56M9tXmmZueytMAGitbxQkjG1TJc5tDaMWrQTRjshe0U3i4uPXoutm9nWmitC4C+4DlSmFwpzBSEDZwqQ4X7jRVORnrN1n3waT2Y/wCCIpLHVqQCLJIyLTW40Vps2IaFtojAUdcQajHfS/uty8IgXDqaJ6Vl2CoN5N6rvazmbIDElC6KS8EuIY1j6us2D/EeyyWgA22w8Ccq0pRdR4VLjBq51SXOpwAoKU6Kn2q0iGx1aWYgryLHtI7lqf8AB8R38drsGxCW8nVPtRHf9j5/pXSocVEQ4q0jG+ij0JFiISaWli2sTCT0JyK9CcoUJLZopQpbNFKYThrw7b1fjxg7H4z6/wB7gfVe4NK81/UrYZY7/wAtg8ji0PA/a7fydQX76/zBZ/JOxv8A42pnXL+uG2iyrTWlKH1yVdItrBaDkS09E9MxCXWa3Xn0qFqCPJTie9Fjnx4dWp56hKsIcHZ5a3Lp5E3Ai7gfod3NUcuwZqyl6uuGCn7cP69WzopNwpXfUe6BNsaxtp5/KsZGVa1to5Xrn9qvL4gLjRuVcBhROUfWdLywtxAcguwlLgFzclEhl4DHsJ3A39l1MpDrROQWyQadeGttWauwJAvp9Ukx4cA4XjIgqxjzUIeU3kZDjvXPzDzCfaB8jzhuJPsrvhnOVdQAaUvTsFlErKvBAT7CiJ1BpwW4D20vs1HMEEeyu/D8ENhk5ucSebQGH1aVUQSKEf0nnhiFfbGh2YLB/T7klaT2x3eTh5QiqahFVxjfRN6EivQiqS2sWLEBtyE5FchOUKEls0ZBls0YphgQdpSTY8F8FwFHsLb8iflPMGh6IyIxA6+cpmXcXFjiWuhmlMwQbx0KZhM8ld5JXRfqBs90OdiOs0bEsvacjVoDutprq8xvVJDb5Kbly2ctj0Jrsl/rcvCqr7ZsACiqpYXK72eblC++Gtv7RbCYBmdXrzraviAvNG4bwrXxqHudcTStmnIEn6qlltmQaD4jyCa1ABu48cPVbYzOdrn3rXeZIw555dUE1rdrJdMzxPGZCI+KbQIb+2tKG/D8oclIywNQHuvH7RuoruT2bLA1+CXV3kfdXbCmNuZO3nkVFbW/EOz81b8dXrcLbzwfMDZcMLy2/MbjwwXbMlILvKyW5HygC+txFSq7bWxWBlSyzfUUIcRXKgAu+wT7CuNTz1a+E9p22lpN49QcF1rHLy/w8x8N5IyvPEE0OuBXo8rFqAeCzs5eLl7O1by94PI+xXSyLC2G1pxDRVUGxRV45j0vXTrXPpz7vnjFCIpqERVGd9FHoKM9CTSxYsWJhtyE5GcguUKEls0YoMtiUYpwqxTaoBTCCcl+o8QfAa0Q2viEucytxbZArZI3ktF93ZeUbOnS8va9th38pywFL+NO69K8ZTFqZDB/8bGjq6rj6Fq5Pauxw/ztueM/5huKjWeuj49fWSFZUVCuJB1FTSESpo65wxrvVuDQVC5+crs72ENosxbStTXBVzZJu5WUy60apdhoi0ciMHZzP5QriTgsaPlHVKMJpcm4UKl57lEFXEu9mTW+iZewOyCQYzBMsJVxnqEJqRa0ktAoReKYHhzoE3IXCiY+HUUWtnwS54aMzRMu+HUeHYOLir5LScEMaGjIf5TK2k5HJq/a9YoRFNQi4Jpvoo9CRXoapLVVi2sQGyguRigvUKElsSjFVz9ow4dbRqf5W3n7DqVUTO3YkR1lg+G3M4upzwb0v4pfaH9bXQTM4yH87w2uAxJ5NF6HB2vAcQ34gBOAdVteRcKdFxUd5tmtTvJvJ5kpfbMAuguc3ENJol29Vcznsbaj7cxGd/W5o5NNkeyEIblGQhGyN+ZTohFVCqh2nssnzs+cZZO/KXlpm0LJuIuIONV0roSrNobKL/O254zyPAqNY61+P5OeKRdCQjKmtQtw4zgbDxQjFHZFIWXPyuj7fsZLwrwSE+wE0G5Dl44zH0TJiNyRwfY0KUFMwjQRVICPkjNfkn1NPxIoaBv3q48NMhutOa4F4upurn9Fyk0+8DPEo8hEcx4ew0I1Q7wrzfLLXbOPR4aKktnTQiNtC45jcU6tXPGkONgiIcVAvoo9QU3qAVJaWLaxABmppkMVe4DcMSeQF5XPTu2HPq1gLW7/AN57fL075Kic973FznEk5k1KsJCWBJxqKUOuiy81v9c59iMhki5uGNyixhAvAaOWKelo9o2HYjAqcSXq5rqYJ/UvsrnwCRaA/KARcWnMU5blcvZSoGSrZpmYyT5wregSjMk6AkoES+isGqoiglqgWJqi2Wgpl1VzWzmRB5gQcnC4jW5UE7DiS587C9n/ANjcBuDhl7FdrCg14AYncpRLJaWkCzShabwRnaGdVnrErTPyXLgP/UG4hWsq8OFRmo7d2JDDTEhNsWaWm1NkgkNq2uF5F2Ha9DZYcw2T8pw4FYall5XVnU1Ox0UOXuqiQ4QRYL6tUIrqBIKyJFtPdzp2uTku6hquflpsPcbPzDEHtVXcB+StnV5IzjmOBbWtQKb12kCJaAOq5rjdlQqm2RQC5uOOZFd3uRuXV7N+TqVtn0w1Z3kNqEVTUIuCab6JvQ0V6EmltYsWJhxb5YMurV3o2v1TUhLuAJpjcddfRBjudUjC8n1/BT2z5yovxoe9/wCFEnlrb2F5qWJ8zTRww/KsdnxbbRX5hcRyR2wWvaTmdXqsaTCiA5G481fpHT0yyjq61+FXTMDW/VyvojLbajMeqQdDrd21rBFglUD4BBwVhLGovxHqmjByI1oeiNAgNHvr0SkFpdkAu+UE8gVFxYx1lzr9zb6c3YD1Vq6u8nmTT8Jeckmvvwdv38/VPiekHxXi5zQGnAt+X/PO+5SEMOwN2PZYy0zyuw3H7ayR2S4N7P7c/wDidYoCEWVa+G9jsHAjiLiDTuOy4iC0glrsWmh5g3r0JjTSjscDzVVPeH2PtPY2j73AgkVONCML8K0zWfyY7PDb4vkmbyq2A7ypWfjUYeKyDGFQ3gkZ+JXlguZ1KbZEq90yQwXXl5yDTfjzp2Xey7KkDGgAH1oq7ZMqIcNt3niAPccwDe0dAR1JV5AFgAgcuWZ1xW+M+HPvXnppkTzHgSBwANwCvtlvq08z9FSuYH+ZvzZ8fymZJ5oac+1x+i24w75XyhFwVe2bcM9ar2RP/LJGAI4Y9kuC1j0MKQiB2BUQmTdFimsQHJwojYzdz/ff1xQ4cMtdTWq07JGhY6oNxvHVXMCK2IKOudv38+N/qpWYhOLcNaFnupzMMRGnf/nDssYKXHHWuiIGUvGtAeqtNb2LMEiw7EXc+PW5TmoVDrX+Es4hrw8XE489/PFWxaHtR+F3lJNbaHEeqHEZTWuaKBZOtcUw+FUIOlIUTIpoN17pB7aGhTUtEyTiK3Egh2Pf2ST4DmHC7X49VbOYh1yPEpHC8OIHfN3R7FyXmZez5m9tcgFGDNUuOskFxyG25b4Ud5/a+j29bnD+4O6EKnJtuazeQD1K6zxfBrCDx+w1/wCLqA+tk8L1yey224jevahXLvPN8d3x67jv8dbIwC926t54NGvVX75QECl2X2UdnSthl48zqE8Nw+vVNjXL8rpzORya12khLFuCnAq14NOf1TjXKZYMVXEdIzDLLiMsRyOHt6FQZEv1rRTk8yrbQHy+x+xp0qqyutaxSXPMP2Q68Gjt4z5rVtzT5vwUGDETobVMhrKxQscSsU8DhpYWmCv7fJ2w9FNrS03Ya+6hsvzNe3cWuHWod6AI7N2q6I7KYurCTnK+V+G/XCqtGt66+/sues01rIDuriRjZHWvMVUTQZ9hAqNY09b09syYqBropzEG0NclWSBLHOYRShuHA6PdP9HuLmYh31Rf2gqMN1QiU8pTT0rPwLrQ1uSUN2aunNqKKmissPI7JEsYLrQWojMeyXlIlDROvCYLtN9NYKsnIdHXZ4e3u70VqW0PdITQr6e6Rwo9rXscx97XAg8nVHtZ7lU/g3YL2zD7dCxguOT6kFpplheFZRnUbz+wp7K58Mt8r3b307NB+qjWZbK0zqyWf05ECDa17fdMxhjq5Ku1rkrjNthRwUu3WuSO3X0TTRWbjyKppiFYcW9uIy1xKtmuQtoQrTbQxb7a9ylVZqrYda1eFayT6i9VQGtc/Xgn4VzBzr7JRWlj8NYhfGWJ+U+Hn2xX/wASz/MxzfZ3sCrBzb9azHZU8o+zEYf6gDyd5XehKvYmPH6/5cOyiNaxrbvX6+wb3RoN2un0PdRha+ns3ui2aa1uPdUlbyz7QocdfVJTgsvaf+J61+tO63KvIOtYk9kfacO0y0OY55FMv0SWfrXNNwyqyUfUA9U+wpoNKu2gy+qsCbknOZJQVXsxVtCdabVVVNa5p6QdimQrgq+YZ9B61Vm9qSmWfjmg45+aN7enehXQ+Gj/AA3Dc8+rW/lUE2L+vtQe5KtvDcSjojOTuxIPuFK76XcUZpGK1Pncl3CoIOITiC7UYFBbcphyqFUwUaGa3JcBGhhFEV0eDZcRqmq9SmIfyDmfojTcOoBzHt/miC0Ubrd+Eoq3w3aWlFYqS4KZh3a5q7iPtUcP3AO/uFfd/olJmDcev2+6lBcbDOVn+0loWMb09Bd+Pp/17ptjRrW5p7pGA6t/X6//AJ7p+GctbvZp7qomtlpGtZuKsZc2mlpz0PRKlt2scfcjsiy5odcB9CnCpSWFlzmbj6G/3qrJh16/UJWfZR7XjB1WnmLx7FMMOtck4imobskrMnLWr0Vjqa1xUZhl9d+CCJBtda4J2XbQa5KDGa1yRW67JgwL0tMsuR2FbiMqEg5eeZedcvX2UtiPpGH9QcPS1/1TG0IV+tf5SMqbEVh3Pb2Jofcqb7X7jsECLc4Hf9EYocYVCaC0dtFuAKqUwKtRJdlGhV3wGwxTAWLApDTihPZddwI11Kk8363Ib41lobnd90z6hYWKVViouufiMFnp90rCYLDuDnU/tb9ytLFk2ZLm/W/8BWsIXDp/0+5WlicKnJa/HWKkMun1WliaU5//AEuRaRzqFkPXosWJwqmNdkxE+UcwsWJ1KAw1uWOx1wWLEBKHruUcYLFiQV+02BUM3cbssOmC2sSqsuvehuwKxYhIKbh56yWLEU4kWhDIWLEoKC/Ec/okZj5z09gsWKoQqxYsVE//2Q==")
    print(f"Overall time {time.time() -overall_start}")
    if result=="Certain Error has occured. Check logs":
        print("Check nohup.out . Something went wrong")
    else:
        print(result)

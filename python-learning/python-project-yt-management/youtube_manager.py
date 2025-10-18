import json
def load_data():
    try:
        with open('youtube.txt','r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    

def save_data_helper(videos):
    with open('youtube.txt', 'w')  as file:
        json.dump(videos,file)
    
def list_all_videos(videos):
    for i, video in enumerate(videos,start=1):
        print(f"{i}: name:{video['name']} , duration: {video['time']}")
def add_video(videos):
    name=input("Enter youtube video name: ")
    time=input("Enter youtube video time: ")
    videos.append({"name": name, "time":time})
    save_data_helper(videos)

def update_video(videos):
    list_all_videos(videos)
    index=int (input("Enter update youtube video number: "))
   
    if 1<=index<=len(videos):
        updated_name=input("Enter update youtube video name: ")
        updated_time=input("Enter updated youtube video time: ") 
        videos[index-1]={"name": updated_name, "time":updated_time}
        save_data_helper(videos)
    else:
        print("Invalid index selected!")    

def delete_video(videos):
    list_all_videos(videos)
    index=int (input("Enter deleted youtube video number: "))
    if 1<=index<=len(videos):
        del videos[index-1]
        save_data_helper(videos)
    else:
        print("Invalid index selected!")

def main():
    videos=load_data()
    while True:
        print("\n Youtube manager| chose an option")
        print("1. List all youtube videos")
        print("2. Add a youtube video ")
        print("3.Update a youtube video details ")
        print("4. Delete a youtube video ")
        print("5. Exit the app ")
        choice =input("Enter your choice: ")
        # print(videos)

        match choice:
            case '1':
                list_all_videos(videos)
            case '2':
                add_video(videos)    
            case '3':
                update_video(videos)
            case '4':
                delete_video(videos)
            case '5':
                break
            case _:
                print("Invalid choice")



if __name__=="__main__":
    main()




from tkinter import*
from tkinter import messagebox
from PIL import ImageTk
import pymysql
class REGISTER:
    def __init__(self,root):
        self.root=root
        self.root.title("Login System")
        self.root.geometry("1366x768")
        self.root.config(bg="#021e2f")
        self.root.resizable(False,False)
        Frame_left=Frame(self.root,bg="#08A3D2",bd=0)
        Frame_left.place(x=0,y=0,relheight=1,width=600)



        Frame_login=Frame(self.root,bg="white")
        Frame_login.place(x=250,y=100,height=500,width=800)


        title=Label(Frame_login,text="Registration Form",font=("times new roman",30,"bold"),bg="white",fg="red").place(x=90,y=30)
#=======================Email ID================
        lbl_user=Label(Frame_login,text="Email Address",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=90,y=110)
        self.txt_user=Entry(Frame_login,font=("times new roman",15),bg="gray68")
        self.txt_user.place(x=90,y=150,width=350,height=35)

#=======================Password ================        
        lbl_pass1=Label(Frame_login,text="Password",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=90,y=190)
        self.txt_pass1=Entry(Frame_login,show="*",font=("times new roman",15),bg="gray68")
        self.txt_pass1.place(x=90,y=230,width=350,height=35)

#=======================Confirm Password================
        lbl_cpass=Label(Frame_login,text="Confirm Password",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=90,y=270)
        self.txt_cpass1=Entry(Frame_login,show="*",font=("times new roman",15),bg="gray68")
        self.txt_cpass1.place(x=90,y=310,width=300,height=30)

#========================Mobile Number================
        lbl_contact=Label(Frame_login,text="Contact No.",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=90,y=350)
        self.txt_contact=Entry(Frame_login,font=("times new roman",15),bg="gray68")
        self.txt_contact.place(x=90,y=390,width=300,height=30)
#=======================Button================
        But_reg=Button(Frame_login,text="Register",font=("times new roman",14),bg="white",bd=0,fg="#B00857",command=self.Register_data).place(x=90,y=440)
        
        But_login=Button(Frame_login,text="Login",font=("times new roman",14),bg="white",bd=0,fg="#B00857",command=self.login_window).place(x=190,y=440)
#==========================================================================    
    def login_window(self):
        self.root.destroy()
        import login
        
    def clear(self):
        self.txt_user.delete(0,END)
        self.txt_pass1.delete(0,END)
        self.txt_contact.delete(0,END)
        
        
        
    def Register_data(self):
        
        #print(self.txt_user.get(),self.txt_contact.get(),self.txt_pass1.get())
        if self.txt_user.get()=="" or self.txt_pass1.get()=="" or self.txt_cpass1.get()=="" or self.txt_contact.get()=="":
            messagebox.showerror("Error","All Fields Are Required",parent=self.root)
            
        elif self.txt_pass1.get()!=self.txt_cpass1.get():
            messagebox.showerror("Error","Password And Confirm Password Should Be Same",parent=self.root)
            
        
    
        else:
            try:
                mydb = pymysql.connect(
                host="localhost",
                user="root",
                passwd="",
                database="test1"
                )
                mycursor = mydb.cursor()
                
                mycursor.execute("INSERT INTO user(username,Password,contact) VALUES(%s,%s,%s)",(self.txt_user.get(),self.txt_pass1.get(),self.txt_contact.get()))
                
                mydb.commit()
                mydb.close()
                messagebox.showinfo("Sucess","Registered Succesfully",parent=self.root)
                self.clear()
            except Exception as es:
                messagebox.showerror("Error",f" Error Due To: {str(es)}",parent=self.root)
            self.login_window()


        
root=Tk()
obj=REGISTER(root)
root.mainloop()

def register(x,y=2):
    return x + y
def password(x,y=2):
    return x - y
def login(x,y=2):
    return x * y

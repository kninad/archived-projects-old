using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
using Photon.Pun;


public class CharacterMovement : MonoBehaviourPun
{
    public CharacterController controller;
    private GameObject camera;
    public Canvas canvas;
    public float speed = 1.0f;
    public float gravity = 9.8f;

    private float ySpeed = 0.0f;

    private GameManager gameManager;

    private Vector3 velocity;

    public bool InLegoMode = false;
    public GameObject CurrentLegoObject;
    private float LegoDistance;
    public bool ButtonDown;
    private bool RotateButtonDown;
    private float LastRotateTime;

    private LegoManager legoManager;

    bool isShowed = false;

    public GameObject characterModel;

    public Canvas playerName;
    


    private MenuMode currentMode = MenuMode.Create; // default


    // Start is called before the first frame update
    void Start()
    {
        camera = GameObject.Find("Main Camera");
        playerName.GetComponentInChildren<Text>().text = photonView.Owner.NickName;


        if (PhotonNetwork.IsConnected == true && photonView.IsMine == false)
        {
            return;
        }

        GetComponent<CameraMove>().OnStartFollowing();

        gameManager = FindObjectOfType<GameManager>();
        gameManager.LocalPlayerInstance = this.gameObject;

        
        //camera.transform.SetParent(controller.transform);

        //canvas = GetComponentInChildren<Canvas>();
        //canvas.transform.SetParent(controller.transform);
        canvas.enabled = false; // default, disable the menu.

        
        legoManager = FindObjectOfType<LegoManager>();
        canvas.GetComponent<InventoryMenuControl>().lm = legoManager;

        characterModel.GetComponentInChildren<MeshRenderer>().enabled = false;
        playerName.enabled = false;
    }

    // Update is called once per frame
    void Update()
    {
        Quaternion nameRotate = Quaternion.LookRotation(camera.transform.forward, Vector3.up);
        playerName.transform.rotation = nameRotate;
        

        if (PhotonNetwork.IsConnected == true && photonView.IsMine == false)
        {
            return;
        }

        currentMode = InventoryMenuControl.currMode;

        // Check Exit Game status?
        if (currentMode == MenuMode.ExitGame)
        {
            // return to default after exiting so that user can join back to the same room
            // currentMode = MenuMode.Create;
            InventoryMenuControl.currMode = MenuMode.Create;
            gameManager.LeaveRoom();
        }


        if (Input.GetButtonUp("Submit"))
        {
            ButtonDown = false;
        }

        //if(!InLegoMode)
        //{
        // Find the camera forward and right vectors, without their y components

        Vector3 cameraForward = camera.transform.forward;
        cameraForward.y = 0.0f;
        cameraForward.Normalize();

        Vector3 cameraRight = camera.transform.right;
        cameraRight.y = 0.0f;
        cameraRight.Normalize();

        // Move based on the keyboard input and camera direction
        Vector3 moveDirection = (Input.GetAxis("Horizontal") * cameraRight + Input.GetAxis("Vertical") * cameraForward) * speed;

        ySpeed -= gravity * Time.deltaTime;
        moveDirection.y = ySpeed;

        controller.Move(moveDirection * Time.deltaTime);

        controller.enabled = false;
        // limit to grid bounds
        if(transform.position.x > 100)
        {
            transform.position = new Vector3(100, transform.position.y, transform.position.z);
        }
        if (transform.position.z > 100)
        {
            transform.position = new Vector3(transform.position.x, transform.position.y, 100);
        }
        if (transform.position.x < 0)
        {
            transform.position = new Vector3(0, transform.position.y, transform.position.z);
        }
        if (transform.position.z < 0)
        {
            transform.position = new Vector3(transform.position.x, transform.position.y, 0);
        }
        controller.enabled = true;


        //}
        //else
        if (InLegoMode) {
            Vector3 trueCameraForward = camera.transform.forward;
            
            if (Input.GetButtonUp("Fire1"))
            {
                RotateButtonDown = false;
            }

            if (Input.GetButtonDown("Fire1") && !RotateButtonDown /*&& LastRotateTime + 1f < Time.time*/)
            {
                //LastRotateTime = Time.time;
                RotateButtonDown = true;
                CurrentLegoObject.transform.Rotate(Vector3.up * 90);

            }

            CurrentLegoObject.transform.position = transform.position + trueCameraForward * LegoDistance;
            bool positionValid = CurrentLegoObject.GetComponent<LegoMove>().UpdatePosition();

            // @nak keyboard controls for quick testing. leave them commented out.
            // if (!ButtonDown && positionValid && (Input.GetButtonDown("Submit") || Input.GetKeyDown("enter")))
            if (!ButtonDown && positionValid && Input.GetButtonDown("Submit"))
            {
                ButtonDown = true;
                InLegoMode = false;
                legoManager.TellEveryoneIHaveNoBricks();
            }
        }
        
        controller.Move(moveDirection * Time.deltaTime * speed);

        Quaternion characterRotate = Quaternion.LookRotation(cameraForward, Vector3.up);
        characterModel.transform.rotation = characterRotate;

        ToggleMenu(cameraForward);

    }

    private void ToggleMenu(Vector3 cameraForward)
    {
        // Fire1 is detecting as Mouse Click creating conflicts. So setting it as a
        // Keyboard control for now.
        //if (Input.GetButtonDown("Fire1") && !isShowed)
        if (Input.GetButton("Fire2") && !isShowed && !InLegoMode)
        {
            isShowed = true;
            canvas.enabled = true;
            Quaternion cameraRotate = Quaternion.LookRotation(cameraForward, Vector3.up);
            canvas.transform.position = transform.position + cameraForward * 7.0f;
            canvas.transform.rotation = cameraRotate;

        } else if (InventoryMenuControl.toExit && isShowed)
        {
            Fade();
        }
    }

    // close the menu and reset bools
    private void Fade()
    {
        isShowed = false;
        InventoryMenuControl.toExit = false;
        canvas.transform.position = new Vector3(10000, 10000, 10000);
        canvas.enabled = false;
    }

    public void EnterLegoMode(GameObject legoObject, float distance = 0f)
    {

        InLegoMode = true;
        if(distance == 0)
        {
            ButtonDown = true;
        }
        
        CurrentLegoObject = legoObject;

        // We can predefine the distance when creating new bricks
        if(distance > 0)
        {
            LegoDistance = distance;
        }
        else
        {
            LegoDistance = Vector3.Distance(CurrentLegoObject.transform.position, transform.position);
        }
    }

    bool Vector3InGrid(Vector3 vector)
    {
        return vector.x >= 0
            && vector.x < 100
            && vector.y >= 0
            && vector.y < 100
            && vector.z >= 0
            && vector.z < 100;
    }
}
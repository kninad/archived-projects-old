using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Photon.Pun;
using Photon.Realtime;

public class LegoMove : MonoBehaviour
{
    private LegoManager legoManager;
    private Outline outline;
    private GameManager gameManager;

    //private int XUnits = 1;
    //private int YUnits = 1; // This is multiplied by 1.2
    //private int ZUnits = 1;

    public Vector3Int dimensions;
    public int color;

    private int LegoId;
    private bool HasPointer;
    private bool HasDoneIntialUpdate = false;
    //public bool InitializeInLegoMode;

    public Player InitialPlayer;

    // Start is called before the first frame update
    void Start()
    {
        // make name unique
        //gameObject.name = GetInstanceID().ToString();

        

        Initialize();
    }

    public void Initialize()
    {
        legoManager = FindObjectOfType<LegoManager>();

        // Create the actual lego model, as a child of this object
        //GameObject model = Instantiate(Resources.Load<GameObject>(modelName), transform);
        //model.transform.position = this.transform.position;
        outline = GetComponentInChildren<Outline>();

        outline.OutlineWidth = 8;

        // set up collision box
        //BoxCollider collider = GetComponent<BoxCollider>();
        //collider.size = new Vector3(
        //    dimensions.x,
        //    dimensions.y * 1.2f,
        //    dimensions.z
        //);
        //collider.center = new Vector3(
        //    (dimensions.x - 1) * -0.5f,
        //    (dimensions.y - 1) * -0.5f,
        //    (dimensions.z - 1) * -0.5f
        //);

        string name = "Lego" + dimensions.z + "x" + dimensions.x;
        if(color == 0)
        {
            name += "Blue";
        }
        else if(color == 1)
        {
            name += "Pink";
        }
        else if(color == 2)
        {
            name += "Yellow";
        }


        Vector3Int position = new Vector3Int((int)transform.position.x,
            (int)(transform.position.y / 1.2f), (int)transform.position.z);

        Debug.Log(position);

        LegoId = legoManager.RegisterBrick(dimensions, position, (int)transform.eulerAngles.y, name, this.gameObject);
        
        outline.OutlineColor = new Color(0, 0, 0, 0);

        gameManager = FindObjectOfType<GameManager>();

        
    }

    // Update is called once per frame
    void Update()
    {
        legoManager = FindObjectOfType<LegoManager>();
        MenuMode currentMode = InventoryMenuControl.currMode;

        if(!gameManager)
        {
            // Lego has not be Initialized();
            return;
        }

        GameObject characterObject = gameManager.LocalPlayerInstance;
        if(!characterObject)
        {
            // Local character has not been created
            return;
        }

        CharacterMovement character = characterObject.GetComponent<CharacterMovement>();

        
        if(!HasDoneIntialUpdate && legoManager != null)
        {
            HasDoneIntialUpdate = true;
            UpdatePosition();
        }

        // DELETE BRICK action
        // @nak keyboard control for quick testing: Pressing "D" key deletes the lego.        
        // if((HasPointer && Input.GetKeyDown("d")) && !character.InLegoMode && !character.ButtonDown && (currentMode == MenuMode.Destroy))
        if((HasPointer && Input.GetButtonDown("Submit")) && !character.InLegoMode && !character.ButtonDown && (currentMode == MenuMode.Destroy))
        {
            Debug.Log("Deleting Lego: " + LegoId);            
            legoManager.DeleteBrick(LegoId, this.gameObject);
        }

        // ENTERING LEGO MODE
        if(((HasPointer && Input.GetButtonDown("Submit"))) && !character.InLegoMode && !character.ButtonDown && (currentMode == MenuMode.Create))
        {
            //Debug.Log(legoManager.CanMoveBrick(gameObject));
            if (legoManager.CanMoveBrick(gameObject))
            {
                GetComponent<PhotonView>().TransferOwnership(PhotonNetwork.LocalPlayer);

                legoManager.TellEveryoneAboutMyBrick(gameObject);

                // Automatic distance based on player distance for moving existing block
                character.EnterLegoMode(this.gameObject);
            }
        }

        if (HasPointer && !character.InLegoMode)
        {
            if (currentMode == MenuMode.Destroy)
            {
                outline.OutlineColor = Color.red;
            }
            else if (currentMode == MenuMode.Create)
            {
                outline.OutlineColor = Color.blue;
            }
        }
        else if(!character.InLegoMode)
        {
            outline.OutlineColor = new Color(0, 0, 0, 0);
        }

        //UpdatePosition();
    }

    // Snap to grid
    // Returns whether position is valid
    public bool UpdatePosition()
    {
        // This can be called before the legoManger gets set
        if(legoManager == null)
        {
            return false;
        }

        //Debug.Log("Update pos " + LegoId);

        (Vector3 position, bool validPosition) = legoManager.UpdatePositionBrick(LegoId,
            transform.position, (int)transform.eulerAngles.y);

        transform.position = position;

        if (validPosition)
        {
            outline.OutlineColor = Color.green;
        }
        else
        {
            outline.OutlineColor = Color.red;
        }

        return validPosition;
    }

    public void PointerEnter()
    {
        HasPointer = true;
    }

    public void PointerExit()
    {
        HasPointer = false;
    }

    
}

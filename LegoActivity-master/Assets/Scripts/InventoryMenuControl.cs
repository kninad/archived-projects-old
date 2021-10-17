using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Photon.Pun;


public enum MenuMode {Create, Destroy, ClearAction, LegoMode, ExitGame};


public class InventoryMenuControl : MonoBehaviour
{
    public static MenuMode currMode = MenuMode.Create; // default
    public static bool toExit = false; 
    public LegoManager lm;
    public PhotonView photonView;


    //public GameObject textDisplay;

    void Start()
    {
        SetCreate();
    }

    public void SetCreate()
    {
        currMode = MenuMode.Create;
        //textDisplay.GetComponent<Inventory.ModeText>().text = "Currnt Mode: Create Mode";
        //toExit = true;
        DisplayBuildMode.modeTexttxt = "Create Mode";
    }

    public void SetDestroy()
    {
        currMode = MenuMode.Destroy;
        toExit = true;
        DisplayBuildMode.modeTexttxt = "Destroy Mode";
    }

    public void SetClearAction()
    {
        currMode = MenuMode.ClearAction;
        toExit = true;
    }

    public void SetExit()
    {
        toExit = true;
    }
    //*************************Blue*******************************

    public void Create_1x1_brickBlue()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x1Blue");
            toExit = true;
        }
    }

    public void Create_1x2_brickBlue()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x2", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x2Blue");
            toExit = true;
        }
    }

    public void Create_2x2_brickBlue()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego2x2", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego2x2Blue");
            toExit = true;
        }
    }

    public void Create_1x4_brickBlue()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x4Blue");
            toExit = true;
        }
    }

    

    public void Create_2x4_brickBlue()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego2x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego2x4Blue");
            toExit = true;
        }
    }

    //***************************************************************

    //*************************Pink*******************************

    public void Create_1x1_brickPink()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x1Pink");
            toExit = true;
        }
    }

    public void Create_1x2_brickPink()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x2", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x2Pink");
            toExit = true;
        }
    }

    public void Create_2x2_brickPink()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego2x2", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego2x2Pink");
            toExit = true;
        }
    }

    public void Create_1x4_brickPink()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x4Pink");
            toExit = true;
        }
    }



    public void Create_2x4_brickPink()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego2x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego2x4Pink");
            toExit = true;
        }
    }

    //***************************************************************

    //*************************Yellow*******************************

    public void Create_1x1_brickYellow()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x1Yellow");
            toExit = true;
        }
    }

    public void Create_1x2_brickYellow()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x2", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x2Yellow");
            toExit = true;
        }
    }

    public void Create_2x2_brickYellow()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego2x2", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego2x2Yellow");
            toExit = true;
        }
    }

    public void Create_1x4_brickYellow()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego1x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego1x4Yellow");
            toExit = true;
        }
    }



    public void Create_2x4_brickYellow()
    {
        SetCreate();
        if (currMode == MenuMode.Create)
        {
            //lm.CreateBrick("Lego2x4", new Vector3(1f, 0, 1f));
            lm.AskMasterToCreateBrick("Lego2x4Yellow");
            toExit = true;
        }
    }

    //***************************************************************

    public void SetExitGame()
    {
        currMode = MenuMode.ExitGame;
        toExit = true;
    }

}
